import copy
from math import log
from typing import List, Set, Tuple, Dict
from bisect import bisect_left


class IntDictionary:
    def __init__(self, seqs: List[Tuple[int]] = None):
        self.parents = [-1] * len(seqs) if seqs is not None else []
        if seqs is None:
            self.seqs = []
            return

        self.seqs = copy.deepcopy(seqs)
        parents_stack = []
        self.seqs = sorted(self.seqs)

        for i, current in enumerate(self.seqs):
            while len(parents_stack) > 0:
                prefix = parents_stack[-1][0]
                if current[:len(prefix)] == prefix:
                    self.parents[i] = parents_stack[-1][1]
                    break
                parents_stack.pop()

            parents_stack.append((current, i))

    def search(self, seq: Tuple[int], excludes: Set[int] = None) -> int:
        index = bisect_left(self.seqs, tuple(seq))
        if 0 <= index < self.size() and self.seqs[index] == seq:
            if excludes is None or index not in excludes:
                return index

        index -= 1

        while 0 <= index < self.size():
            found_seq = self.seqs[index]
            if seq[:len(found_seq)] == found_seq:
                if excludes is None or index not in excludes:
                    return index
            index = self.parents[index]

        if excludes is None:
            ind = self.size()
            self.seqs.append(tuple([seq[0]]))
            self.parents.append(-1)
            return ind
        else:
            return -1

    def linear_parse(self, seq: List[int], excludes: Set[int] = None) -> List[int]:
        coded_seq = []
        while len(seq) > 0:
            symbol = self.search(tuple(seq), excludes)
            if symbol < 0:
                seq = seq[1:]
            else:
                sym_len = len(self.seqs[symbol])
                seq = seq[sym_len:]
                coded_seq.append(symbol)

        return coded_seq

    def weighted_parse(self, seq: List[int], freqs: List[int],
                       total_freq: float, excludes: Set[int] = None) -> List[int]:
        l = len(seq)
        score = [-float("inf")] * (l + 1)
        score[0] = 0.0
        symbols = [-1] * (l + 1)

        for pos in range(l):
            suffix = seq[pos:]
            sym = self.search(tuple(suffix), excludes)
            if sym < 0:
                continue

            while sym >= 0:
                sym_len = len(self.get(sym))
                sym_log_prob = (log(freqs[sym] + 1) if sym < len(freqs) else 0) - log(total_freq + self.size())
                if score[sym_len + pos] < score[pos] + sym_log_prob:
                    score[sym_len + pos] = score[pos] + sym_log_prob
                    symbols[sym_len + pos] = sym
                sym = self.parent(sym)

        solution = [0] * (l + 1)
        pos = l
        index = 0
        while pos > 0:
            index += 1
            sym = symbols[pos]
            solution[l - index] = sym
            pos -= len(self.get(sym))

        return solution[l - index:-1]

    # def _parse(self, seq: List[int], freqs: List[int], total_freq: float) -> List[int]:
    #     return self.weighted_parse(seq, freqs, total_freq, None)

    def parse(self, seq: List[int], excludes: Set[int] = None) -> List[int]:
        return self.linear_parse(seq, excludes)

    def get(self, ind: int) -> Tuple[int]:
        return self.seqs[ind]

    def size(self):
        return len(self.seqs)

    def parent(self, ind: int) -> int:
        return self.parents[ind]

    def alphabet(self) -> List[Tuple[int]]:
        return self.seqs

    def __eq__(self, other):
        if not isinstance(other, IntDictionary):
            return False
        return self.seqs == other.seqs and self.parents == other.parents

    def to_json(self) -> Dict:
        return {"seqs": self.seqs}

    @staticmethod
    def from_json(json) -> 'IntDictionary':
        return IntDictionary(json["seqs"])
