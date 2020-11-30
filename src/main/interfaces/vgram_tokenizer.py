import json
import random
from typing import List, Union, Optional

from main.algorithm.int_dictionary import IntDictionary
from main.algorithm.vgram_builder import VGramBuilder
from main.interfaces.coder import SimpleCoder
from main.interfaces.tokenizer import Tokenizer


class VGramTokenizer(Tokenizer):
    def __init__(self, size: int = 30000, words_level: bool = True):
        self.coder = SimpleCoder()
        self.size = size
        self.words_level = words_level
        self.dict: Optional[IntDictionary] = None

    def _split_words(self, text: str) -> List[str]:
        if self.words_level:
            words = []
            word = ""
            for c in text:
                if not c.isalnum():
                    words.append(word)
                    word = ""
                word += c
            words.append(word)
        else:
            words = [text]
        return words

    def _encode_one(self, seq: str) -> List[int]:
        coded = []
        for word in self._split_words(seq):
            coded += self.dict.parse(self.coder.encode(word))
        return coded

    def encode(self, seqs: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        if type(seqs) is str:
            return self._encode_one(seqs)
        return [self._encode_one(seq) for seq in seqs]

    def tokenize(self, seqs: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        def tokenize_one(seq: str) -> List[str]:
            coded = self._encode_one(seq)
            return [self.coder.decode(self.dict.get(id)) for id in coded]

        if type(seqs) is str:
            return tokenize_one(seqs)
        return [tokenize_one(seq) for seq in seqs]

    def decode(self, coded_seqs: Union[int, List[int], List[List[int]]]) -> Union[str, List[str]]:
        def decode_one(seq: List[int]) -> str:
            return "".join([self.coder.decode(self.dict.get(id)) for id in seq])

        if type(coded_seqs) is int:
            return decode_one([coded_seqs])

        assert len(coded_seqs) > 0
        if type(coded_seqs[0]) is int:
            return decode_one(coded_seqs)
        return [decode_one(seq) for seq in coded_seqs]

    def fit(self, texts: Union[str, List[str]], iters: int = 1, verbose: int = 0):
        vgram_builder = VGramBuilder(self.size, verbose)
        if type(texts) is str:
            texts = [texts]
        for iter in range(iters):
            for i in range(len(texts)):
                line = texts[random.randint(0, len(texts) - 1)]
                for word in self._split_words(line):
                    ids = self.coder.encode(word)
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
                        line = lines[i].strip()
                        for word in self._split_words(line):
                            ids = self.coder.encode(word)
                            vgram_builder.accept(ids)

        self.coder.fix()
        self.dict = vgram_builder.result()

    def get_vocab(self) -> List[str]:
        return [self.coder.decode(list(seq)) for seq in self.dict.alphabet()]

    def vocab_size(self) -> int:
        return self.dict.size()

    def save_pretrained(self, path: str):
        res = {"dict": self.dict.to_json(), "coder": self.coder.to_json(),
               "words_level": self.words_level, "size": self.size}
        json.dump(res, open(path, 'w'))

    @staticmethod
    def from_pretrained(path: str) -> 'VGramTokenizer':
        res = json.load(open(path))
        tokenizer = VGramTokenizer(res["size"], res["words_level"])
        tokenizer.dict = IntDictionary.from_json(res["dict"])
        tokenizer.coder = SimpleCoder.from_json(res["coder"])
        return tokenizer
