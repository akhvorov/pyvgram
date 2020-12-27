import copy
import numpy as np
from math import log
from typing import List, Tuple, Set

from vgram.main.algorithm.int_dictionary import IntDictionary


class StatItem:
    def __init__(self, first: int, second: int, score: float, count: int):
        self.first = first
        self.second = second
        self.score = score
        self.count = count

    def __eq__(self, other: 'StatItem'):
        return self.first == other.first and self.second == other.second


class StatDictionary:
    kMaxMinProbability = 0.002
    kAggPower = 100000

    def __init__(self, min_prob_result: float = kMaxMinProbability,
                 seqs: List[Tuple[int]] = None,
                 init_freqs: List[int] = None):
        self.dict = IntDictionary(seqs)
        self.symbol_freqs = []
        self.parse_freqs = []

        if seqs is not None:
            self.symbol_freqs = [0] * len(seqs)

            if init_freqs is None:
                self.parse_freqs = [0] * len(seqs)
            else:
                _, init_freqs = zip(*sorted(zip(seqs, init_freqs)))
                self.parse_freqs = copy.deepcopy(list(init_freqs))
            self.parse_freqs += [0] * (len(seqs) - len(self.parse_freqs))

        self.pairs_freqs = {}
        self.min_probability = min_prob_result

        self.power = 0
        self.parse_freqs_init_power = sum(self.parse_freqs)
        self.total_chars = 0

    def update_symbol(self, index: int, freq: int):
        if index >= len(self.symbol_freqs):
            for i in range(len(self.symbol_freqs), index + 1):
                self.symbol_freqs.append(0)
            for i in range(len(self.parse_freqs), index + 1):
                self.parse_freqs.append(0)

        self.symbol_freqs[index] += freq
        self.parse_freqs[index] += freq
        self.power += freq

    def search(self, seq: Tuple[int], excludes: Set[int] = None) -> int:
        return self.dict.search(seq, excludes)

    def parse(self, seq: List[int]) -> List[int]:
        parse_result = self.dict.weighted_parse(seq, self.parse_freqs, self.power + self.parse_freqs_init_power)

        self.total_chars += len(seq)
        prev = -1
        for symbol in parse_result:
            self.update_symbol(symbol, 1)
            if prev >= 0:
                self.pairs_freqs[(prev, symbol)] = self.pairs_freqs.get((prev, symbol), 0) + 1
            prev = symbol

        return parse_result

    def get(self, ind: int) -> Tuple[int]:
        return self.dict.get(ind)

    def size(self):
        return self.dict.size()

    def parent(self, ind: int) -> int:
        return self.dict.parent(ind)

    def alphabet(self) -> List[Tuple[int]]:
        return self.dict.alphabet()

    def freq(self, index: int) -> int:
        return self.symbol_freqs[index] if index < len(self.symbol_freqs) else 0

    def code_length_per_char(self):
        summa = 0
        for i in range(self.size()):
            frequency = self.freq(i)
            if frequency > 0:
                summa -= frequency * log(frequency)
        return (summa + self.power * log(self.power)) / self.total_chars

    def enough(self, prob_found: float) -> bool:
        return self.power > -log(prob_found) / self.min_probability

    def expand(self, slots: int) -> 'StatDictionary':
        new_dict = []
        freqs = []

        items = []
        known = set()
        for seq in self.alphabet():
            known.add(seq)
            symbol = self.search(seq)
            items.append(StatItem(-1, symbol, float("inf"), self.freq(symbol)))

        slots += len(self.alphabet())  # TODO why?
        start_with = [0] * len(self.symbol_freqs)
        ends_with = [0] * len(self.symbol_freqs)
        for pair, freq in self.pairs_freqs.items():
            start_with[pair[0]] += freq
            ends_with[pair[1]] += freq

        total_pair_freqs = sum(start_with)
        for pair, freq in self.pairs_freqs.items():
            ab = freq
            xb = ends_with[pair[1]] - freq
            ay = start_with[pair[0]] - freq
            xy = total_pair_freqs - ay - xb - ab

            dirichlet_params = (ab + 1, ay + 1, xb + 1, xy + 1)
            score = 0
            samples_count = 10
            for i in range(samples_count):
                sample = np.random.dirichlet(dirichlet_params)
                pAB = sample[0]
                pAY = sample[1]
                pXB = sample[2]
                score += freq * pAB / (pAY + pAB) * log(pAB / (pAY + pAB) / (pXB + pAB)) / samples_count

            item = StatItem(pair[0], pair[1], score, freq)
            item_text = self.stat_item_to_text(item)
            if item_text not in known:
                known.add(item_text)
                items.append(item)

        items = sorted(items, key=lambda x: -x.score)
        min_prob_result = self.min_probability
        accumulated_freqs = sum(self.pairs_freqs.values())

        for item in items:
            if item.score < 0:
                break
            slots -= 1
            if slots < 0:
                break

            item_text = self.stat_item_to_text(item)
            new_dict.append(item_text)
            freqs.append(item.count)
            if item.first >= 0:
                min_prob_result = min(min_prob_result, item.count / accumulated_freqs)

        return StatDictionary(min_prob_result, new_dict, freqs)

    def reduce(self, slots: int) -> 'StatDictionary':
        new_dict = []
        freqs = []
        items = self.filter_stat_items(slots)
        power = 0.0
        for item in items:
            power += item.count

        min_prob_result = min(1. / self.size(), self.min_probability)
        for item in items:
            p = (item.count + 1.0) / (power + self.size())
            if self.parent(item.second) >= 0:
                min_prob_result = min(p, min_prob_result)
            new_dict.append(self.get(item.second))
            freqs.append(item.count)

        return StatDictionary(min_prob_result, new_dict, freqs)

    def filter_stat_items(self, slots: int) -> List[StatItem]:
        excludes = set()
        for id in range(self.size()):
            if self.parent(id) >= 0 and self.freq(id) == 0:
                excludes.add(id)

        items = self.stat_items(excludes)
        while len(items) > min(len(items), slots):
            items.pop()

        return items

    def stat_items(self, excludes: Set[int]):
        items = []
        code_length = self.code_length_per_char() * self.total_chars
        for id in range(len(self.symbol_freqs)):
            if id not in excludes:
                count = self.freq(id)
                seq = self.get(id)

                if len(seq) > 1:
                    excludes.add(id)
                    parse = self.dict.weighted_parse(list(seq), self.symbol_freqs, self.power, excludes)  # TODO comment from c++ code: doesn't work, make base_dict
                    excludes.remove(id)

                    new_power = self.power + (len(parse) - 1) * count
                    code_length_without_symbol = code_length - self.power * log(self.power) + new_power * log(new_power)
                    if count != 0:
                        code_length_without_symbol += count * log(count)
                    for next_id in parse:
                        old_freq = self.freq(next_id)
                        new_freq = old_freq + count
                        if count > 0:
                            code_length_without_symbol -= new_freq * log(new_freq) - \
                                                          (old_freq * log(old_freq) if old_freq > 0 else 0)

                    score = code_length_without_symbol - code_length
                    if score > 0:
                        items.append(StatItem(-1, id, score, count))
                else:
                    items.append(StatItem(-1, id, float("inf"), count))

        items = sorted(items, key=lambda x: -x.score)
        return items

    def result_freqs(self) -> List[int]:
        if self.size() > len(self.parse_freqs):
            for i in range(len(self.parse_freqs), self.size()):
                self.parse_freqs.append(0)
        return copy.deepcopy(self.parse_freqs)

    def stat_item_to_text(self, item: 'StatItem') -> Tuple[int]:
        if item.first >= 0:
            return tuple(list(self.get(item.first)) + list(self.get(item.second)))
        else:
            return self.get(item.second)
