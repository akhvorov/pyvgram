from abc import ABC
from typing import List

from vgram.main.interfaces.tokenizer import Tokenizer


class BaseTokenizer(Tokenizer, ABC):
    def __init__(self, words_level: bool = True):
        self.words_level = words_level

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
