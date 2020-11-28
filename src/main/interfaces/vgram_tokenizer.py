import random
from typing import List, Union

from main.algorithm.int_dictionary import IntDictionary
from main.algorithm.vgram_builder import VGramBuilder
from main.interfaces.coder import SimpleCoder
from main.interfaces.tokenizer import Tokenizer


class VGramTokenizer(Tokenizer):
    def __init__(self, size: int = 30000):
        self.coder = SimpleCoder()
        self.size = size
        self.dict: IntDictionary = None

    def encode(self, seqs: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        def encode_one(seq: str) -> List[int]:
            return self.dict.parse(self.coder.encode(seq))

        if type(seqs) is str:
            return encode_one(seqs)
        return [encode_one(seq) for seq in seqs]

    def tokenize(self, seqs: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        def tokenize_one(seq: str) -> List[str]:
            coded = self.dict.parse(self.coder.encode(seq))
            return [self.coder.decode(self.dict.get(id)) for id in coded]

        if type(seqs) is str:
            return tokenize_one(seqs)
        return [tokenize_one(seq) for seq in seqs]

    def decode(self, coded_seqs: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        def decode_one(seq: List[int]) -> str:
            return "".join([self.coder.decode(self.dict.get(id)) for id in seq])

        assert len(coded_seqs) > 0

        if type(coded_seqs[0]) is int:
            return decode_one(coded_seqs)
        return [decode_one(seq) for seq in coded_seqs]

    def save_pretrained(self, path: str):
        raise NotImplementedError

    def fit(self, texts: Union[str, List[str]], iters: int = 1, verbose: int = 0):
        vgram_builder = VGramBuilder(self.size, verbose)
        if type(texts) is str:
            texts = [texts]
        for iter in range(iters):
            for i in range(len(texts)):
                line = texts[random.randint(0, len(texts))]
                ids = self.coder.encode(line)
                vgram_builder.accept(ids)

        self.dict = vgram_builder.result()

    def train(self, files: Union[str, List[str]], iters: int = 1, verbose: int = 0):
        vgram_builder = VGramBuilder(self.size, verbose)
        if type(files) is str:
            files = [files]
        for iter in range(iters):
            for file in files:
                with open(file) as f:
                    lines = f.readlines()
                    for i in range(len(lines)):
                        # line = lines[random.randint(0, len(lines))]
                        line = lines[i]
                        ids = self.coder.encode(line)
                        vgram_builder.accept(ids)

        self.coder.fix()
        self.dict = vgram_builder.result()

    def get_vocab(self) -> List[str]:
        return [self.coder.decode(list(seq)) for seq in self.dict.alphabet()]

    def vocab_size(self) -> int:
        return self.dict.size()

    @staticmethod
    def from_pretrained(path: str) -> 'Tokenizer':
        raise NotImplementedError