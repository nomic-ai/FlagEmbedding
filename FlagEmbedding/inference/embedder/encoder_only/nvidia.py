import os
import torch
import numpy as np
from typing import List, Dict, Optional, Union
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from FlagEmbedding.abc.inference import AbsEmbedder
from transformers.configuration_utils import PretrainedConfig


class NvidiaEmbedderConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name_or_path = "nvidia"


class NvidiaMockModel:
    def __init__(self):
        self.config = NvidiaEmbedderConfig()


class NvidiaEmbedder(AbsEmbedder):
    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        query_instruction_for_retrieval: Optional[str] = None,
        query_instruction_format: str = "{}{}",
        devices: Optional[Union[str, List[str]]] = None,
        batch_size: int = 2048,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        convert_to_numpy: bool = True,
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
        
        self.model = NvidiaMockModel()
        self.client = OpenAI(
            api_key="not-needed",  # API key not needed for local server
            base_url="http://localhost:8000/v1"
        )
        self.model_name = "nvidia/llama-3.2-nv-embedqa-1b-v2"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        reraise=True
    )
    def _get_embeddings(self, texts: List[str], input_type: str = "query") -> np.ndarray:
        """Get embeddings for a batch of texts with automatic retries and truncation on token size errors."""
        def try_with_texts(current_texts: List[str], retry_count: int = 0) -> np.ndarray:
            try:
                response = self.client.embeddings.create(
                    input=current_texts,
                    model=self.model_name,
                    encoding_format="float",
                    extra_body={"input_type": input_type, "truncate": "END"}
                )
                return np.array([data.embedding for data in response.data])
            except Exception as e:
                error_str = str(e)
                print(f"Error in _get_embeddings: {error_str}")
                
                # If we hit token size limit and haven't retried too many times, truncate and retry
                if "token size" in error_str.lower() and retry_count < 3:
                    truncated_texts = [t[:len(t)//2] for t in current_texts]
                    print(f"Retrying with truncated texts (retry {retry_count + 1})")
                    return try_with_texts(truncated_texts, retry_count + 1)
                
                raise
        
        return try_with_texts(texts)

    def encode_queries(self, queries: List[str], batch_size: int = 128, **kwargs) -> np.ndarray:
        """Encode queries with input_type='query'."""
        all_embeddings = []
        for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
            batch = queries[i:i + batch_size]
            embeddings = self._get_embeddings(batch, input_type="query")
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    @staticmethod
    def _process_batch_static(args):
        """Static method to process a batch of passages."""
        index, batch, model_name, base_url = args
        try:
            # Create a new client for each process
            client = OpenAI(
                api_key="not-needed",
                base_url=base_url
            )
            
            # Get embeddings
            response = client.embeddings.create(
                input=batch,
                model=model_name,
                encoding_format="float",
                extra_body={"input_type": "passage", "truncate": "END"}
            )
            embeddings = np.array([data.embedding for data in response.data])
            return index, embeddings
        except Exception as e:
            print(f"Error processing batch {index}: {str(e)}")
            return index, None

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 128, num_processes: int = 16, **kwargs) -> np.ndarray:
        """Encode corpus passages with input_type='passage' using multiple processes."""
        if isinstance(corpus[0], dict):
            passages = [f"{doc.get('title', '')} {doc['text']}".strip() for doc in corpus]
        else:
            passages = corpus

        # Prepare batches with their indices and required parameters
        batches = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i + batch_size]
            # Include model name and base URL for each batch
            batches.append((len(batches), batch, self.model_name, self.client.base_url))

        # Process batches in parallel
        from multiprocessing import Pool
        with Pool(processes=num_processes) as pool:
            # Use tqdm to show progress
            results = list(tqdm(
                pool.imap(self._process_batch_static, batches),
                total=len(batches),
                desc=f"Encoding corpus with {num_processes} processes"
            ))

        # Sort results by index and collect embeddings
        sorted_results = sorted(results, key=lambda x: x[0])
        all_embeddings = []
        for _, embeddings in sorted_results:
            if embeddings is not None:
                all_embeddings.append(embeddings)
            else:
                raise Exception("One or more batches failed to process")

        return np.vstack(all_embeddings)

    @torch.no_grad()
    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 128,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
    ):
        """Single device encoding method that defaults to query encoding."""
        if isinstance(sentences, str):
            sentences = [sentences]
        return self.encode_queries(sentences, batch_size=batch_size)
