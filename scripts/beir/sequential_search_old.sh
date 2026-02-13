#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

source /opt/conda/etc/profile.d/conda.sh

retriever_name=$1
nickname=$2
base_model=$3

# if base model is empty, use meta-llama/Llama-2-7b-hf
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
    'fiqa'
    'nfcorpus'
    'scidocs'
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
    # if the final eval file exists and has a score, skip
    if [[ -f "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval" ]]; then
        # if there exists an ndcg_cut_10 in the file or a recip_rank, skip
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

    # if the last command failed, continue to next
    if [ $? -ne 0 ]; then
        echo "Failed to search ${dataset_name}${output_suffix}"
        return
    fi

    echo "Ranking is saved at ${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.txt"

    python -m tevatron.utils.format.convert_result_to_trec \
    --input "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.txt" \
    --output "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
    --remove_query

    # if msmarco is not in the name use beir
    echo "Evaluating ${dataset_name}${output_suffix}..."
    if [[ "$dataset_name" != *"msmarco"* ]] && [[ "$dataset_name" != *"-dev"* ]]; then
        conda activate pyserini && python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 \
        "beir-v1.0.0-${dataset_name}-test" \
        "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
        > "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
    # else if -dev in the name
    elif [[ "$dataset_name" == *"-dev"* ]] && [[ "$dataset_name" != *"msmarco"* ]]; then
        # remove the -dev
        new_dataset_name=$(echo $dataset_name | sed 's/-dev//')
        conda activate pyserini && python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 \
        resources/downloaded/qrels/$new_dataset_name.qrels.sampled \
        "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
        > "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
    else
        dataset=$(echo $dataset_name | cut -d'-' -f2)
        if [ $dataset == "dev" ]; then
            echo "Evaluating ${dataset}..."
            conda activate pyserini && python -m pyserini.eval.trec_eval -c -M 100 -m recip_rank msmarco-passage-dev-subset \
            "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
            > "${nickname}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
        else
            pyserini_dataset="${dataset}-passage"
            echo "Evaluating ${dataset}..."
            conda activate pyserini && python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 $pyserini_dataset \
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

    # Encode corpus
    missing=0
    for i in {0..15}
    do
        if [ ! -f "$dataset_path/corpus_emb_${i}.pkl" ]; then
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

    # Encode queries
    echo "Encoding queries for $dataset..."
    bash encode_beir_queries.sh "$base_model" "$dataset_path" "$retriever_name" "$dataset" "0"
    if [ $? -ne 0 ]; then
        echo "Failed to encode queries for $dataset"
        continue
    fi

    # Encode queries with prompts if generic_prompts.csv exists
    if [ -f "generic_prompts.csv" ]; then
        echo "Encoding queries with prompts for $dataset..."
        while IFS= read -r prompt
        do
            bash encode_beir_queries.sh "$base_model" "$dataset_path" "$retriever_name" "$dataset" "0" "$prompt"
        done < generic_prompts.csv
    fi

    # Search and evaluate
    echo "Searching and evaluating $dataset..."

    # Search without prompt
    if [[ -f "${dataset_path}/${dataset}_queries_emb.pkl" ]]; then
        search_and_evaluate "$dataset" "${dataset_path}/${dataset}_queries_emb.pkl" ""
    fi

    # Search with generic prompts
    for query_file in "${dataset_path}/${dataset}_queries_emb_"*.pkl; do
        if [[ -f "$query_file" ]]; then
            prompt_hash=$(basename "$query_file" | sed -n 's/.*_emb_\(.*\)\.pkl/\1/p')
            search_and_evaluate "$dataset" "$query_file" "_${prompt_hash}"
        fi
    done

    # Clean up pickle files to save disk space
    echo "Cleaning up pickle files for $dataset..."

    # Remove corpus embeddings
    rm -f "$dataset_path"/*.pkl
    echo "$dataset_path"
    echo "Cleanup completed for $dataset"

    # Log completion
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

    # Process results without prompt
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

    # Process results with prompts
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