from abc import ABC
from typing import Optional, List, Iterable, Dict, Tuple

from vgram.main.algorithm.int_dictionary import IntDictionary
from vgram.main.algorithm.vgram_builder import VGramBuilder


class VGramApplier(ABC):
    def __init__(self, size: int = 30000, err_msg: str = None):
        self.size = size
        self.err_msg = err_msg

        self.dict: Optional[IntDictionary] = None
        self.freqs: Optional[List[int]] = None
        self.total_freqs = 0

    def get(self, ind: int) -> Tuple[int]:
        return self.dict.get(ind)

    def parse(self, seq: List[int]) -> List[int]:
        if self.dict is None:
            raise ValueError(self.err_msg)
        return self.dict.weighted_parse(seq, self.freqs, self.total_freqs)

    def __eq__(self, other):
        if not isinstance(other, VGramApplier):
            return False
        return self.dict == self.dict and self.freqs == self.freqs and self.total_freqs == self.total_freqs

    def to_json(self) -> Dict:
        return {
            "size": self.size,
            "seqs": self.dict.seqs,
            "freqs": self.freqs,
        }


class StaticVGramApplier(VGramApplier):
    def __init__(self, size: int = 30000):
        super().__init__(size, "Applier is not fitted")

    def parse(self, seq: List[int]) -> List[int]:
        return self.dict.weighted_parse(seq, self.freqs, self.total_freqs)

    def fit(self, seqs: Iterable[List[int]], verbose: bool = False):
        vgram_builder = VGramBuilder(self.size, verbose)
        for seq in seqs:
            vgram_builder.accept(seq)

        self.dict = vgram_builder.result()
        self.freqs = vgram_builder.result_freqs()
        self.total_freqs = sum(self.freqs)

    @staticmethod
    def from_json(json) -> 'VGramApplier':
        applier = StaticVGramApplier(json["size"])
        seqs = [tuple(s) for s in json["seqs"]]
        applier.dict = IntDictionary(seqs)
        applier.freqs = json["freqs"]
        applier.total_freqs = sum(applier.freqs)
        return applier


class IterativeVGramApplier(VGramApplier):
    def __init__(self, size: int = 30000, verbose: bool = False):
        super().__init__(size, "Applier is not updated. Please call `update()` before parsing")
        self.vgram_builder = VGramBuilder(self.size, verbose)

    def parse(self, seq: List[int]) -> List[int]:
        if self.dict is None:
            raise ValueError("Applier is not updated. Please call `update()` before parsing")
        return self.dict.weighted_parse(seq, self.freqs, self.total_freqs)

    def accept(self, seq: List[int]):
        self.vgram_builder.accept(seq)

    def update(self):
        self.dict = self.vgram_builder.result()
        self.freqs = self.vgram_builder.result_freqs()
        self.total_freqs = sum(self.freqs)
