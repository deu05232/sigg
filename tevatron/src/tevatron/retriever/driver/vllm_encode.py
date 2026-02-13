import logging
import os
import pickle
import sys
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import EncodeDataset
from tevatron.retriever.collator import VllmEncodeCollator
from vllm import LLM
from vllm.config import PoolerConfig
from vllm.inputs import token_inputs
from vllm.lora.request import LoRARequest

from itertools import islice

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    # tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
        cache_dir=model_args.cache_dir
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    if training_args.bf16:
        torch_dtype = 'bfloat16'
    elif training_args.fp16:
        torch_dtype = 'float16'
    else:
        torch_dtype = 'float32'

    if model_args.pooling.lower() == 'eos':
        pooling_type = "LAST"
    else:
        pooling_type = model_args.pooling.upper()
        
    pooler_config = PoolerConfig(pooling_type=pooling_type,
                                 normalize=model_args.normalize)

    model = LLM(
        model=model_args.model_name_or_path,
        tokenizer=tokenizer_name,
        task="embed",
        enforce_eager=True,
        override_pooler_config=pooler_config,
        dtype=torch_dtype,
        enable_lora=True if model_args.lora_name_or_path else False,
        max_lora_rank=model_args.lora_r
    )

    encode_dataset = EncodeDataset(
        data_args=data_args,
    )

    encode_collator = VllmEncodeCollator(
        data_args=data_args,
        tokenizer=tokenizer,
    )

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    
    def chunk_iterable(iterable, num_chunks):
        iterable = list(iterable)
        total = len(iterable)
        chunk_sizes = [(total + i) // num_chunks for i in range(num_chunks)]  # 분배량
        chunks = []
        start = 0
        for size in chunk_sizes:
            end = start + size
            chunks.append(iterable[start:end])
            start = end
        return chunks
    
    if "queries" in data_args.encode_output_path or len(encode_loader) < 100 or data_args.encode_is_query:
        lookup_indices = []
        vllm_inputs = []
        for (batch_ids, batch) in tqdm(encode_loader, desc="Preprocessing"):
            lookup_indices.extend(batch_ids)
            vllm_inputs.extend([token_inputs(prompt_token_ids=token_ids) for token_ids in batch])

        outputs = model.embed(vllm_inputs,
                            lora_request=LoRARequest("emb_adapter",
                                                    1,
                                                    model_args.lora_name_or_path) if model_args.lora_name_or_path else None)

        encoded = []
        for output in outputs:
            encoded.append(output.outputs.embedding)
        encoded = np.stack(encoded, dtype=np.float16)

        with open(data_args.encode_output_path, 'wb') as f:
            pickle.dump((encoded, lookup_indices), f)
        
    else:
        all_batches = list(encode_loader)
        chunks = list(chunk_iterable(all_batches, 8))   # 4개로 나누어서 진행
        
        for i, chunk in enumerate(chunks):
            lookup_indices = []
            vllm_inputs = []
            encoded = []

            for (batch_ids, batch) in tqdm(chunk, desc=f"Processing chunk {i+1}/8"):
                lookup_indices.extend(batch_ids)
                vllm_inputs.extend([token_inputs(prompt_token_ids=token_ids) for token_ids in batch])
                
            # 해당 부분에서, lora checkpoint에 tokenizer가 없으면 default tokenizer로 설정됨에 주의
            # tokenizer 파일을 넣어주어야 함;
            outputs = model.embed(
                vllm_inputs,
                lora_request=LoRARequest("emb_adapter", 1, model_args.lora_name_or_path)
                if model_args.lora_name_or_path else None
            )

            for output in outputs:
                encoded.append(output.outputs.embedding)
            encoded = np.stack(encoded, dtype=np.float16)
            
            output_path = data_args.encode_output_path.replace(".pkl", f"_{i}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump((encoded, lookup_indices), f)

            del encoded, lookup_indices
        



if __name__ == "__main__":
    main()
