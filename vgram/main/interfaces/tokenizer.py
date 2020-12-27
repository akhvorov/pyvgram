from abc import ABC, abstractmethod
from typing import List, Union


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, seqs: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, seqs: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, coded_seqs: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        raise NotImplementedError

    @abstractmethod
    def save_pretrained(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def train(self, files: Union[str, List[str]], iters: int = 1):
        raise NotImplementedError

    @abstractmethod
    def get_vocab(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @staticmethod
    def from_pretrained(path: str) -> 'Tokenizer':
        raise NotImplementedError
