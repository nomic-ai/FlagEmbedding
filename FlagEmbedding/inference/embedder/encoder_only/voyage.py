import time
import voyageai
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
from functools import partial
from FlagEmbedding.abc.inference import AbsEmbedder
import hashlib
import json
import pathlib
import logging

VOYAGE_API_KEY = "pa-pL_OHgl4hBg7rAtZVtAgVTg3LNqhPoqJT0fI3UNBlpR"


def split_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i : i + chunk_size]

from transformers.configuration_utils import PretrainedConfig
class VoyageEmbedderConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name_or_path = "voyage"

class VoyageMockModel():
    def __init__(self):
        self.config = VoyageEmbedderConfig()

MODEL = 'voyage-3-large'

def call_api_query(text_chunk, cache_dir="./voyage_cache"):
    """Simple function that just calls the API - used in multiprocessing"""
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    # Create cache key from input
    chunks_str = json.dumps(text_chunk, sort_keys=True)
    cache_key = hashlib.md5(chunks_str.encode()).hexdigest()
    cache_file = cache_dir / f"query_{cache_key}.npy"
    
    # Check cache first
    if cache_file.exists():
        try:
            return np.load(cache_file).tolist()
        except Exception as e:
            print(f"Failed to load cache file {cache_file}: {e}")
    
    # If not in cache, call API
    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = vo.embed(text_chunk, model=MODEL, input_type='query')
            embeddings = result.embeddings
            # Save to cache
            np.save(cache_file, embeddings)
            return embeddings
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                raise e
            time.sleep(5)

def call_api_document(text_chunk, cache_dir="./voyage_cache"):
    """Simple function that just calls the API - used in multiprocessing"""
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    # Create cache key from input
    chunks_str = json.dumps(text_chunk, sort_keys=True)
    cache_key = hashlib.md5(chunks_str.encode()).hexdigest()
    cache_file = cache_dir / f"doc_{cache_key}.npy"
    
    # Check cache first
    if cache_file.exists():
        try:
            return np.load(cache_file).tolist()
        except Exception as e:
            print(f"Failed to load cache file {cache_file}: {e}")
    
    # If not in cache, call API
    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = vo.embed(text_chunk, model=MODEL, input_type='document')
            embeddings = result.embeddings
            # Save to cache
            np.save(cache_file, embeddings)
            return embeddings
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                raise e
            time.sleep(5)


class VoyageEmbedder(AbsEmbedder):
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

        self.encoder_batch_size = 64
        self.vo = voyageai.Client(api_key=VOYAGE_API_KEY)
        self.pool = None
        self.config = VoyageEmbedderConfig()
        self.embedding_model = 'voyage-3-large'
        cache_dir = "./voyage_cache"
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else pathlib.Path("voyage_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

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

        self.model = VoyageMockModel()
        """
        Initialize RetrievalModel with the same parameters as BaseEmbedder.
        """

    @torch.no_grad()
    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
    ):
        return self.encode_queries(sentences, batch_size=batch_size)


    def encode_queries(self, queries: List[str], batch_size: int = 128, num_parallel=32, **kwargs) -> np.ndarray:
        print('Encoding queries')
        # Prepare chunks outside of multiprocessing
        chunks = list(split_list(queries, batch_size))
        
        # Setup the pool and process chunks
        with mp.Pool(num_parallel) as pool:
            results = list(tqdm(
                pool.imap(call_api_query, chunks),
                desc="Encoding queries",
                total=len(chunks)
            ))
        
        # Flatten results
        total_encoded_queries = []
        for result in results:
            total_encoded_queries.extend(result)
        
        return np.array(total_encoded_queries)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 128, num_parallel=32, **kwargs) -> np.ndarray:
        print('Encoding corpus')
        # Prepare passages outside of multiprocessing
        if isinstance(corpus[0], dict):
            passages = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            passages = corpus
        
        passages = [passage[:8192*8] for passage in passages]
        chunks = list(split_list(passages, batch_size))

        with mp.Pool(num_parallel) as pool:
            results = list(tqdm(
                pool.imap(call_api_document, chunks),
                desc="Encoding documents",
                total=len(chunks)
            ))
        
        # Flatten results
        total_encoded_queries = []
        for result in results:
            total_encoded_queries.extend(result)

        
        return np.array(total_encoded_passages)