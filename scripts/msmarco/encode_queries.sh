#!/bin/bash
encoded_save_path=$1
model=$2
model_type=$3

# if model type is empty then use llama2
if [ -z "$model_type" ]
then
  model_type="meta-llama/Llama-2-7b-hf"
fi

echo "Lora rank 32로 설정되어 있음!"
echo "Path to save: $encoded_save_path"
echo "Model and Model Type: $model $model_type"

for dataset in dl19 dl20 dev; do
echo $dataset
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.vllm_encode \
  --output_dir=temp \
  --model_name_or_path $model_type \
  --lora_name_or_path $model \
  --lora \
  --lora_r 32 \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 256 \
  --dataset_name Tevatron/msmarco-passage-new \
  --dataset_split $dataset \
  --encode_output_path $encoded_save_path/${dataset}_queries_emb.pkl 
done
  
