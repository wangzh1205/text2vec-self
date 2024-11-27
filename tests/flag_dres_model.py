# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Evaluate MTEB benchmark

pip install mteb

code modified from https://github.com/FlagOpen/FlagEmbedding
"""
import os
from typing import cast, List, Dict, Union

import numpy as np
import torch
from mteb import DRESModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"


class FlagDRESModel(DRESModel):
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            batch_size: int = 128,
            **kwargs
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        self.batch_size = batch_size

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.batch_size = self.batch_size * num_gpus

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts)

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.encode(input_texts)

    @torch.no_grad()
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        self.model.eval()

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + self.batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = last_hidden_state[:, 0]
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)
