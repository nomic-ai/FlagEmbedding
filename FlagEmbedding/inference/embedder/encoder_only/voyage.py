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

    def encode_queries(self, queries: List[str], batch_size: int = 256, **kwargs) -> np.ndarray:
        
        #queries = [cutoff_long_text_for_embedding_generation(query, self.encoding, cutoff=4096) for query in queries]
        total_encoded_queries = []
        #for query_chunks in tqdm(split_list(queries, self.encoder_batch_size), total=len(queries)//self.encoder_batch_size):
        for query_chunks in tqdm(split_list(queries, batch_size), total=len(queries)//batch_size):
            try:
                encoded_queries = self.vo.embed(query_chunks, model=self.embedding_model, input_type='query')
                encoded_queries = encoded_queries.embeddings
            except Exception as e:
                raise e
                time.sleep(5)
                encoded_queries = self.vo.embed(query_chunks, model=self.embedding_model, input_type='query')
                encoded_queries = encoded_queries.embeddings

            #encoded_queries = [query_encoding for query_encoding in encoded_queries]
            total_encoded_queries += encoded_queries
        return np.array(total_encoded_queries)

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 256, **kwargs) -> np.ndarray:
        if isinstance(corpus[0], dict):
            passages = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            passages = corpus

        passages = [
            passage[:8192*8] #modify for context length
            for passage in passages
        ]
        
        total_encoded_passages = []
        #for passage_chunks in tqdm(split_list(passages, self.encoder_batch_size), total=len(passages)//self.encoder_batch_size):
        for passage_chunks in tqdm(split_list(passages, batch_size), total=len(passages)//batch_size):
            # Create a hash of the passage chunks for the cache filename
            chunks_str = json.dumps(passage_chunks, sort_keys=True)
            cache_key = hashlib.md5(chunks_str.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.npy"

            if cache_file.exists():
                # Load cached embeddings if they exist
                self.logger.info(f"Cache hit for key: {cache_key[:8]}...")
                encoded_passages = np.load(cache_file)
                encoded_passages = encoded_passages.tolist()
            else:
                attempts = 0
                while attempts < 5 and not cache_file.exists():
                    try:
                        encoded_passages = self.vo.embed(passage_chunks, model=self.embedding_model, input_type='document')
                        encoded_passages = encoded_passages.embeddings
                        # Cache the results
                        np.save(cache_file, encoded_passages)
                    except Exception as e:
                        attempts += 1
                        self.logger.warning(f"API call failed for key: {cache_key[:8]}... Retrying after 30s")
                        time.sleep(5)
                if not cache_file.exists():
                    raise Exception(f"Failed to retrieve embeddings after 5 attempts for key: {cache_key[:8]}")

            total_encoded_passages += encoded_passages
        return np.array(total_encoded_passages)