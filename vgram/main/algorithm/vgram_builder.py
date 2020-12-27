import copy
from math import log
from typing import List, Tuple

from vgram.main.algorithm.int_dictionary import IntDictionary
from vgram.main.algorithm.stat_dictionary import StatDictionary


class VGramBuilder:
    def __init__(self, size: int, verbose: bool = True):
        self._size = size
        self._symb_alphabet = []
        self._current = StatDictionary()
        self._result = None
        self._verbose = verbose

        self._populate = True
        self._prob_found = 0.1
        self._best_compression_rate = 1.0
        self._no_rate_increase_turns = 0

        self._kExtensionFactor = 1.3
        self._kMaxPower = 20000000

    def result(self) -> IntDictionary:
        return self._current.dict if self._result is None else self._result.dict

    def alphabet(self) -> List[Tuple[int]]:
        if self._result is None:
            return self._current.alphabet()
        return self._result.alphabet()

    def accept(self, seq: List[int]):
        self._current.parse(seq)
        if self._current.enough(self._prob_found) or self._current.power > self._kMaxPower:
            self._update()

    def _update(self):
        if not (self._current.enough(self._prob_found) or self._current.power > self._kMaxPower):
            return

        sum = 0
        text_length = 0
        for i in range(self._current.size()):
            freq = self._current.freq(i)
            text_length += len(self._current.get(i)) * freq
            if freq > 0:
                sum -= freq * log(freq) / log(2)

        code_length = (sum + self._current.power * log(self._current.power) / log(2)) / 8.
        compression_rate = code_length / text_length
        if compression_rate < self._best_compression_rate:
            self._best_compression_rate = compression_rate
            self._no_rate_increase_turns = 0
        elif self._no_rate_increase_turns > 2:
            self._no_rate_increase_turns += 1
            self._prob_found *= 0.8

        alphabet_size = len(self._symb_alphabet)
        if self._populate:
            if self._verbose > 0 and self._result is not None:
                print(f"Size: {self._result.size()} rate: {compression_rate} minimal probability: {self._current.min_probability}")

            if self._current.size() * self._kExtensionFactor < 10:
                slots = self._size - alphabet_size
            else:
                slots = self._current.size() * self._kExtensionFactor
            self._current = self._current.expand(slots)
        else:
            self._current = self._current.reduce(self._size)
            self._result = copy.deepcopy(self._current)

        alpha_accum_num = 0
        for i in range(self._current.size()):
            if self._current.parent(i) < 0:
                if alpha_accum_num >= alphabet_size:
                    self._symb_alphabet.append(self._current.get(i))
                alpha_accum_num += 1
        self._populate = not self._populate

    def result_freqs(self):
        if self._result is None:
            return self._current.result_freqs()
        return self._result.result_freqs()

    def code_length(self):
        return self._result.code_length_per_char()

    def get_min_probability(self):
        return self._result.min_probability if self._result is not None else -1.0
