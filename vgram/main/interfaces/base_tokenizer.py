from abc import ABC
from typing import List

from vgram.main.interfaces.splitter import SplitLevel
from vgram.main.interfaces.tokenizer import Tokenizer


class BaseTokenizer(Tokenizer, ABC):
    def __init__(self, split_level: SplitLevel = SplitLevel.WORD):
        self.split_level = split_level

    def _split_words(self, text: str, split_level: SplitLevel = None) -> List[str]:
        split_level = split_level or self.split_level

        if split_level == SplitLevel.NONE:
            return [text]
        elif split_level == SplitLevel.LINE:
            lines = text.split('\n')
            tokens = [lines[0]]
            for word in lines[1:]:
                tokens.append('\n')
                tokens.append(word)
            return tokens
        elif split_level == SplitLevel.WORD:
            lines = self._split_words(text, SplitLevel.LINE)
            words = []
            for line in lines:
                word = ""
                for c in line:
                    if not c.isalnum():
                        if word:
                            words.append(word)
                        word = ""
                    word += c
                if word:
                    words.append(word)
            return words
        else:
            raise ValueError("Wrong split level")
