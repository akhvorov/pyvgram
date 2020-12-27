import json
import random
from typing import List, Union

from vgram.main.interfaces.vgram_applier import IterativeVGramApplier, StaticVGramApplier
from vgram.main.interfaces.base_tokenizer import BaseTokenizer
from vgram.main.interfaces.coder import SimpleCoder


class VGramTokenizer(BaseTokenizer):
    def __init__(self, size: int = 30000, words_level: bool = True, verbose: bool = False):
        super().__init__(words_level)
        self.coder = SimpleCoder()
        self.verbose = verbose
        self.vgram_applier = IterativeVGramApplier(size, self.verbose)

    def _encode_one(self, seq: str) -> List[int]:
        coded = []
        for word in self._split_words(seq):
            coded += self.vgram_applier.parse(self.coder.encode(word))
        return coded

    def encode(self, seqs: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        if type(seqs) is str:
            return self._encode_one(seqs)
        return [self._encode_one(seq) for seq in seqs]

    def tokenize(self, seqs: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        def tokenize_one(seq: str) -> List[str]:
            coded = self._encode_one(seq)
            return [self.coder.decode(self.vgram_applier.get(id)) for id in coded]

        if type(seqs) is str:
            return tokenize_one(seqs)
        return [tokenize_one(seq) for seq in seqs]

    def decode(self, coded_seqs: Union[int, List[int], List[List[int]]]) -> Union[str, List[str]]:
        def decode_one(seq: List[int]) -> str:
            return "".join([self.coder.decode(self.vgram_applier.get(id)) for id in seq])

        if type(coded_seqs) is int:
            return decode_one([coded_seqs])

        assert len(coded_seqs) > 0
        if type(coded_seqs[0]) is int:
            return decode_one(coded_seqs)
        return [decode_one(seq) for seq in coded_seqs]

    def fit(self, texts: Union[str, List[str]], iters: int = 1):
        if type(texts) is str:
            texts = [texts]
        for iter in range(iters):
            for i in range(len(texts)):
                line = texts[random.randint(0, len(texts) - 1)]
                for word in self._split_words(line):
                    ids = self.coder.encode(word)
                    self.vgram_applier.accept(ids)

        self.coder.fix()
        self.vgram_applier.update()

    def train(self, files: Union[str, List[str]], iters: int = 1):
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
                            self.vgram_applier.accept(ids)

        self.coder.fix()
        self.vgram_applier.update()

    def get_vocab(self) -> List[str]:
        return [self.coder.decode(list(seq)) for seq in self.vgram_applier.dict.alphabet()]

    def vocab_size(self) -> int:
        return self.vgram_applier.dict.size()

    def __eq__(self, other):
        if not isinstance(other, VGramTokenizer):
            return False
        return self.coder == self.coder and self.vgram_applier == other.vgram_applier

    def save_pretrained(self, path: str):
        res = {"applier": self.vgram_applier.to_json(), "coder": self.coder.to_json(),
               "words_level": self.words_level}
        json.dump(res, open(path, 'w'))

    @staticmethod
    def from_pretrained(path: str) -> 'VGramTokenizer':
        res = json.load(open(path))
        tokenizer = VGramTokenizer(res["words_level"])
        tokenizer.vgram_applier = StaticVGramApplier.from_json(res["applier"])
        tokenizer.coder = SimpleCoder.from_json(res["coder"])
        return tokenizer
