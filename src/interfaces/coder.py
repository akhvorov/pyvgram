from abc import ABC, abstractmethod
from typing import List, Iterable


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
            if not self._fixed and c not in self._char_to_int:
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

    def fix(self):
        self._fixed = True
