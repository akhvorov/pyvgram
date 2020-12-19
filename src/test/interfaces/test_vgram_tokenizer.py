import os
from functools import lru_cache

from main.interfaces.tokenizer import Tokenizer
from main.interfaces.vgram_tokenizer import VGramTokenizer


@lru_cache(maxsize=10)
def train_default_tokenizer(size: int = 1000, iters: int = 1, words_level: bool = False) -> Tokenizer:
    data_dir = "/Users/aleksandr.khvorov/jb/grazie/grazie-datasets/data/"
    files = [data_dir + "stardust/all-texts.txt"]

    tokenizer = VGramTokenizer(size, words_level=words_level, verbose=True)
    tokenizer.train(files, iters=iters)
    return tokenizer


def test_train():
    tokenizer = train_default_tokenizer(1000, 1)
    encoded_seq = tokenizer.encode("hello world")
    print(encoded_seq)
    decoded = tokenizer.decode(encoded_seq)
    assert decoded == "hello world"
    print([tokenizer.decode([i]) for i in tokenizer.encode("fix bug")])


def test_fit():
    tokenizer = VGramTokenizer(200, words_level=False, verbose=True)
    tokenizer.fit(["hello", "hello world"] * 1000, iters=15)
    encoded_seq = tokenizer.encode("hello world")
    assert len(encoded_seq) == 1
    assert len(tokenizer.encode("hello")) == 1
    print(encoded_seq)
    decoded = tokenizer.decode(encoded_seq)
    print(decoded)
    assert decoded == "hello world"


if __name__ == '__main__':
    test_fit()
    # test_train()
    # test_words_level()
    # test_save_and_load()
