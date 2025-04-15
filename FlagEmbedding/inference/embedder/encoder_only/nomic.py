import os
import torch
import torch.multiprocessing as mp
# Set start method to spawn
mp.set_start_method('spawn', force=True)
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from queue import Empty
from typing import List, Dict, Optional, Union
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data.distributed import DistributedSampler
from datasets import Dataset
from contrastors import BiEncoder, BiEncoderConfig
from functools import partial
from FlagEmbedding.abc.inference import AbsEmbedder


def _transform_func(tokenizer,
                    examples: Dict[str, List]):
    return tokenizer(examples['contents'],
                     max_length=512,
                     padding=True,
                     truncation=True)


# Triton is not thread safe AFAICT so using naive DataParallel fails
class EncoderWorker(mp.Process):
    def __init__(self, rank, world_size, input_queue, output_queue, model_name, tokenizer_name, batch_size, master_port=12344):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.master_port = master_port

    def run(self):
        # Initialize process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(self.master_port)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['RANK'] = str(self.rank)
        os.environ['LOCAL_RANK'] = str(self.rank)

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.rank)

        # Initialize model
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        config = BiEncoderConfig.from_pretrained(self.model_name)
        encoder = BiEncoder.from_pretrained(self.model_name, config=config)
        encoder = encoder.to(self.rank)
        encoder = DistributedDataParallel(encoder, device_ids=[self.rank])
        encoder.eval()

        while True:
            try:
                # Get input texts from queue
                input_texts = self.input_queue.get(timeout=60)
                if input_texts is None:  # Poison pill
                    break

                # Process the batch
                dataset = Dataset.from_dict({'contents': input_texts})
                dataset.set_transform(partial(_transform_func, tokenizer))

                # Calculate actual number of samples for this worker
                total_size = len(dataset)
                per_worker = (total_size + self.world_size - 1) // self.world_size
                worker_start = self.rank * per_worker
                worker_end = min(worker_start + per_worker, total_size)
                actual_samples = worker_end - worker_start
                if actual_samples == 0:
                    # create fake work
                    worker_start -= per_worker

                # print(f"Rank {self.rank} - Total size: {total_size}, Start: {worker_start}, End: {worker_end}, Actual samples: {actual_samples}")

                # Create indices for this worker
                indices = list(range(worker_start, worker_end))
                
                # Create a subset of the dataset
                subset = torch.utils.data.Subset(dataset, indices)

                data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
                loader = DataLoader(
                    subset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=data_collator,
                    num_workers=0,
                    pin_memory=True
                )

                local_embeds = []
                with torch.no_grad():
                    for batch_dict in tqdm(loader, desc=f"Rank {self.rank}", disable=True):
                        batch_dict = {k: v.cuda(self.rank) for k, v in batch_dict.items()}
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            outputs = encoder(**batch_dict)
                            local_embeds.append(outputs["embedding"].cpu())

                local_embeds = torch.cat(local_embeds, dim=0)
                
                # Gather embeddings
                # Use actual_samples instead of embedding size for gathering
                # print(f"Rank {self.rank} - Actual samples: {actual_samples}")
                local_size = torch.tensor([actual_samples], device=self.rank)
                # print(f"Rank {self.rank} - Local size: {local_size}")
                all_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
                dist.all_gather(all_sizes, local_size)
                all_sizes = [size.item() for size in all_sizes]
                
                # print(f"Rank {self.rank} max_size: {max(all_sizes)}, all_sizes: {all_sizes}")

                max_size = max(all_sizes)
                padded_embeds = torch.zeros(
                    max_size, local_embeds.shape[1],
                    dtype=local_embeds.dtype, device=self.rank
                )
                padded_embeds[:local_embeds.shape[0]] = local_embeds.cuda(self.rank)

                all_embeds = [torch.zeros_like(padded_embeds) for _ in range(self.world_size)]
                dist.all_gather(all_embeds, padded_embeds)

                if self.rank == 0:  # Only rank 0 returns results
                    result = []
                    for size, embeds in zip(all_sizes, all_embeds):
                        result.append(embeds[:size].cpu().numpy())
                    self.output_queue.put(np.concatenate(result, axis=0))
                
            except Empty:
                continue
            except Exception as e:
                print(f"Worker {self.rank} encountered error: {e}")
                if self.rank == 0:
                    self.output_queue.put(e)
                break

        dist.destroy_process_group()


class NomicEmbedder(AbsEmbedder):
    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        query_instruction_for_retrieval: Optional[str] = None,
        query_instruction_format: str = "{}{}",
        devices: Optional[Union[str, List[str]]] = None,
        pooling_method: str = "cls",
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
        batch_size: int = 256,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        convert_to_numpy: bool = True,
        master_port: int = 29500,
        **kwargs
    ):
        super().__init__(
            model_name_or_path,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
            query_instruction_for_retrieval=query_instruction_for_retrieval,
            query_instruction_format=query_instruction_format,
            devices=devices,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )
        """
        Initialize RetrievalModel with the same parameters as BaseEmbedder.
        """
        self.config = BiEncoderConfig.from_pretrained(
            model_name_or_path,
        )
        self.model = BiEncoder.from_pretrained(
            model_name_or_path,
            config=self.config
        )
        self.world_size = torch.cuda.device_count()
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        
        # Start worker processes
        self.workers = []
        

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """Encode queries using distributed workers."""
        input_texts = queries
        input_texts = [f"search_query: {q}".strip() for q in queries]
        if self.query_instruction_for_retrieval:
            input_texts = [
                self.query_instruction_format.format(self.query_instruction_for_retrieval, q)
                for q in queries
            ]
        return self.encode_single_device(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        """Encode corpus documents using distributed workers."""
        texts = [f"search_document: {doc}".strip() for doc in corpus]
        return self.encode_single_device(texts)

    @torch.no_grad()
    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 512,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # Initialize workers if not already initialized
        if len(self.workers) == 0:
            for rank in range(self.world_size):
                worker = EncoderWorker(
                    rank=rank,
                    world_size=self.world_size,
                    input_queue=self.input_queue,
                    output_queue=self.output_queue,
                    model_name=self.model.config._name_or_path,
                    tokenizer_name="FacebookAI/xlm-roberta-base",
                    batch_size=batch_size,
                )
                worker.start()
                self.workers.append(worker)

        # Calculate number of batches
        total_samples = len(sentences)
        batch_size = 65536
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        all_results = []
        
        # Process sentences in batches
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            batch_sentences = sentences[start_idx:end_idx]
            
            # Distribute batch to workers
            for _ in range(self.world_size):
                self.input_queue.put(batch_sentences)
            
            # Get results for this batch
            batch_result = self.output_queue.get()
            
            if isinstance(batch_result, Exception):
                raise batch_result
            
            all_results.append(batch_result)
        
        # Concatenate results from all batches
        if len(all_results) > 1:
            if isinstance(all_results[0], np.ndarray):
                final_result = np.concatenate(all_results, axis=0)
            else:  # Assuming torch.Tensor
                final_result = torch.cat(all_results, dim=0)
        else:
            final_result = all_results[0]
        
        return final_result

    def __del__(self):
        # Send poison pills to workers
        for _ in range(self.world_size):
            self.input_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join()