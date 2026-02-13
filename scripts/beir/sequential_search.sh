#!/bin/bash
source ~/.bashrc
echo "[DEBUG] HOST=$(hostname) HOME=$HOME USER=$USER"
echo "[DEBUG] proxy env:"
env | grep -i proxy || true
export CUDA_VISIBLE_DEVICES="0,1"

source /opt/conda/etc/profile.d/conda.sh

retriever_name=$1
nickname=$2
base_model=$3

if [ -z "$base_model" ]; then
    base_model="meta-llama/Llama-2-7b-hf"
fi

echo "========================================"
echo "Retriever name: $retriever_name"
echo "Nickname: $nickname"
echo "Base model: $base_model"
echo "========================================"

mkdir -p $nickname
mkdir -p logs/sequential

datasets=(
    #'fiqa'
    #'nfcorpus'
    #'scidocs'
    'scifact'
    'trec-covid'
    'webis-touche2020'
    'quora'
    'nq'
    'arguana'
    'hotpotqa'
    'fever'
    'climate-fever'
    'dbpedia-entity'
)

search_and_evaluate() {
    local dataset_name=$1
    local query_emb_file=$2
    local output_suffix=$3

    conda activate base
    if [[ -f "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval" ]]; then
        if [[ $(grep -c "ndcg_cut_10" "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval") -gt 0 ]] || [[ $(grep -c "recip_rank" "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval") -gt 0 ]]; then
            echo "Skipping ${dataset_name}${output_suffix} because of existing file ${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
            return
        fi
    fi

    echo "Searching and evaluating ${dataset_name} with ${query_emb_file}..."

    python -m tevatron.retriever.driver.search \
    --query_reps "${query_emb_file}" \
    --passage_reps "${nickname}/${dataset_name}/corpus_emb*.pkl" \
    --batch_size 128 \
    --depth 1000 \
    --save_text \
    --save_ranking_to "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.txt"

    if [ $? -ne 0 ]; then
        echo "Failed to search ${dataset_name}${output_suffix}"
        return
    fi

    echo "Ranking is saved at ${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.txt"

    python -m tevatron.utils.format.convert_result_to_trec \
    --input "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.txt" \
    --output "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
    --remove_query

    echo "Evaluating ${dataset_name}${output_suffix}..."

    QRELS_DIR="/workspace/promptriever/scripts/beir/anserini-tools/topics-and-qrels"

    if [[ "$dataset_name" != *"msmarco"* ]] && [[ "$dataset_name" != *"-dev"* ]]; then
        qrels_path="${QRELS_DIR}/qrels.beir-v1.0.0-${dataset_name}.test.txt"

        if [[ ! -f "$qrels_path" ]]; then
            echo "[WARN] Missing qrels file: $qrels_path"
            echo "[WARN] Skipping evaluation for ${dataset_name}${output_suffix}"
            return
        fi

        conda activate pyserini && python -m pyserini.eval.trec_eval -c -m recall.100 -m ndcg_cut.10 \
        "$qrels_path" \
        "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
        > "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"

    elif [[ "$dataset_name" == *"-dev"* ]] && [[ "$dataset_name" != *"msmarco"* ]]; then
        new_dataset_name=$(echo "$dataset_name" | sed 's/-dev//')
        conda activate pyserini && python -m pyserini.eval.trec_eval -c -m recall.100 -m ndcg_cut.10 \
        "resources/downloaded/qrels/${new_dataset_name}.qrels.sampled" \
        "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
        > "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"

    else
        dataset=$(echo "$dataset_name" | cut -d'-' -f2)
        if [ "$dataset" == "dev" ]; then
            echo "Evaluating ${dataset}..."
            conda activate pyserini && python -m pyserini.eval.trec_eval -c -M 100 -m recip_rank msmarco-passage-dev-subset \
            "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
            > "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
        else
            pyserini_dataset="${dataset}-passage"
            echo "Evaluating ${dataset}..."
            conda activate pyserini && python -m pyserini.eval.trec_eval -c -m recall.100 -m ndcg_cut.10 "$pyserini_dataset" \
            "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
            > "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
        fi
    fi

    echo "Score is saved at ${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
    cat "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
    sleep 5
}

# Process each dataset sequentially
for dataset in "${datasets[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing dataset: $dataset"
    echo "========================================"

    dataset_path="$nickname/$dataset"
    mkdir -p "$dataset_path"

    conda activate base

    # Encode corpus (GPU 0, 1 병렬 — 샤드 2개)
    missing=0
    for i in {0..1}
    do
        if [ ! -f "$dataset_path/corpus_emb.${i}.pkl" ]; then
            missing=1
            break
        fi
    done

    if [ $missing -eq 1 ]; then
        echo "Encoding corpus for $dataset (missing embeddings)..."
        bash encode_beir_corpus.sh "$dataset_path" "$retriever_name" "$dataset" "$base_model"
        if [ $? -ne 0 ]; then
            echo "Failed to encode corpus for $dataset"
            continue
        fi
    else
        echo "Corpus embeddings already exist. Skipping corpus encoding."
    fi

    # Encode queries (GPU 0)
    echo "Encoding queries for $dataset..."
    bash encode_beir_queries.sh "$base_model" "$dataset_path" "$retriever_name" "$dataset" "0"
    if [ $? -ne 0 ]; then
        echo "Failed to encode queries for $dataset"
        continue
    fi

    # Encode queries with prompts (GPU 0, 1 번갈아 병렬)
    if [ -f "generic_prompts.csv" ]; then
        echo "Encoding queries with prompts for $dataset..."
        MAX_JOBS=2
        gpu_id=0

        while IFS= read -r prompt; do
            # 빈 줄/공백 줄 스킵(옵션이지만 안전)
            [ -z "$prompt" ] && continue

            # 실행 중인 job 수가 2개 이상이면 하나 끝날 때까지 대기
            while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do
                wait -n
            done

            bash encode_beir_queries.sh "$base_model" "$dataset_path" "$retriever_name" "$dataset" "$gpu_id" "$prompt" &
            gpu_id=$(( (gpu_id + 1) % 2 ))
        done < generic_prompts.csv

        wait
    fi

    # Search and evaluate
    echo "Searching and evaluating $dataset..."

    if [[ -f "${dataset_path}/${dataset}_queries_emb.pkl" ]]; then
        search_and_evaluate "$dataset" "${dataset_path}/${dataset}_queries_emb.pkl" ""
    fi

    for query_file in "${dataset_path}/${dataset}_queries_emb_"*.pkl; do
        if [[ -f "$query_file" ]]; then
            prompt_hash=$(basename "$query_file" | sed -n 's/.*_emb_\(.*\)\.pkl/\1/p')
            search_and_evaluate "$dataset" "$query_file" "_${prompt_hash}"
        fi
    done

    # Clean up
    echo "Cleaning up pickle files for $dataset..."
    rm -f "$dataset_path"/*.pkl
    echo "$dataset_path"
    echo "Cleanup completed for $dataset"

    echo "$(date): Completed $dataset" >> logs/sequential/completion_log.txt
done

echo ""
echo "========================================"
echo "ALL DATASETS PROCESSED!"
echo "========================================"

# Aggregate results
echo "Aggregating results..."
output_file="${nickname}/aggregate_results.csv"
echo "Dataset,Prompt,NDCG@10,Recall@100,MRR" > "$output_file"

for dataset in "${datasets[@]}"; do
    dataset_path="${nickname}/${dataset}"

    eval_file="${dataset_path}/rank.${dataset}.eval"
    if [[ -f "$eval_file" ]]; then
        if [[ "$dataset" == "msmarco"* ]]; then
            mrr=$(awk '/recip_rank / {print $3}' "$eval_file")
            echo "${dataset},no_prompt,,,${mrr}" >> "$output_file"
        else
            ndcg=$(awk '/ndcg_cut_10 / {print $3}' "$eval_file")
            recall=$(awk '/recall_100 / {print $3}' "$eval_file")
            echo "${dataset},no_prompt,${ndcg},${recall}," >> "$output_file"
        fi
    fi

    for eval_file in "${dataset_path}/rank.${dataset}_"*.eval; do
        if [[ -f "$eval_file" ]]; then
            prompt_hash=$(basename "$eval_file" | sed -n 's/.*_\(.*\)\.eval/\1/p')
            if [[ "$dataset" == "msmarco"* ]]; then
                mrr=$(awk '/recip_rank / {print $3}' "$eval_file")
                echo "${dataset},${prompt_hash},,,${mrr}" >> "$output_file"
            else
                ndcg=$(awk '/ndcg_cut_10 / {print $3}' "$eval_file")
                recall=$(awk '/recall_100 / {print $3}' "$eval_file")
                echo "${dataset},${prompt_hash},${ndcg},${recall}," >> "$output_file"
            fi
        fi
    done
done

echo "Aggregate results saved to ${output_file}"
echo ""
echo "========================================"
echo "COMPLETE!"
echo "========================================"