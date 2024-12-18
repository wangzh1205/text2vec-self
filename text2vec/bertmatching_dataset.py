# -*- coding: utf-8 -*-


from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class BertMatchingTrainDataset(Dataset):
    """BERT文本匹配训练数据集, 重写__getitem__和__len__方法
        return: (text, label),
            text format: CLS, sentence1, SEP, sentence2, SEP
            label format: int
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 64):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text_1: str, text_2: str = None):
        return self.tokenizer(text_1, text_2, max_length=self.max_len * 2, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line[0], line[1]), line[2]


class BertMatchingTestDataset(Dataset):
    """BERT文本匹配测试数据集, 重写__getitem__和__len__方法"""

    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 64):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text_1: str, text_2: str = None):
        return self.tokenizer(text_1, text_2, max_length=self.max_len * 2, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line[0], line[1]), line[2]


class HFBertMatchingTrainDataset(Dataset):
    """Load HuggingFace datasets to BERT format

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer
        name (str): dataset name
        max_len (int): max length of sentence
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, name="STS-B", max_len: int = 64):
        self.tokenizer = tokenizer
        self.data = load_dataset("shibing624/nli_zh", name.upper(), split="train")
        self.max_len = max_len
        self.name = name.upper()

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text_1: str, text_2: str = None):
        return self.tokenizer(text_1, text_2, max_length=self.max_len * 2, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        # STS-B convert to 0/1 label
        return self.text_2_id(line['sentence1'], line['sentence2']), int(
            line['label'] > 2.5) if 'STS' in self.name else line['label']


class HFBertMatchingTestDataset(Dataset):
    """Load HuggingFace datasets to Bert format

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer
        name (str): dataset name
        max_len (int): max length of sentence
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, name="STS-B", max_len: int = 64, split="validation"):
        self.tokenizer = tokenizer
        self.data = load_dataset("shibing624/nli_zh", name.upper(), split=split)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text_1: str, text_2: str = None):
        return self.tokenizer(text_1, text_2, max_length=self.max_len * 2, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line['sentence1'], line['sentence2']), line['label']
