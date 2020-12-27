from abc import ABC, abstractmethod
from typing import List, Iterable, Dict


class Coder(ABC):
    @abstractmethod
    def encode(self, seq: str) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, seq: List[int]) -> str:
        raise NotImplementedError


class SimpleCoder(Coder):
    def __init__(self):
        self._char_to_int = {}
        self._int_to_char = {}
        self._fixed = False

    def encode(self, seq: str) -> List[int]:
        result = []
        for c in seq:
            if c not in self._char_to_int:
                if self._fixed:
                    continue
                self._char_to_int[c] = len(self._char_to_int)
                self._int_to_char[self._char_to_int[c]] = c
            result.append(self._char_to_int[c])
        return result

    def decode(self, seq: Iterable[int]) -> str:
        result = ""
        for c in seq:
            result += self._int_to_char[c]
        return result

    def char_to_id(self, c: str) -> int:
        return self._char_to_int[c]

    def id_to_char(self, id: int) -> str:
        return self._int_to_char[id]

    def __len__(self):
        return len(self._char_to_int)

    def __eq__(self, other):
        if not isinstance(other, SimpleCoder):
            return False
        return self._char_to_int == other._char_to_int and \
               self._int_to_char == other._int_to_char and \
               self._fixed == other._fixed

    def fix(self):
        self._fixed = True

    def to_json(self) -> Dict:
        res = {"fixed": self._fixed, "chars": []}
        for i, c in self._int_to_char.items():
            res["chars"].append(c)
        return res

    @staticmethod
    def from_json(json) -> 'Coder':
        coder = SimpleCoder()
        for c in json["chars"]:
            coder.encode(c)
        if json["fixed"]:
            coder.fix()
        return coder
