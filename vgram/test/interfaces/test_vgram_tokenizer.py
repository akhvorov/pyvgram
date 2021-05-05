import os
from functools import lru_cache

from vgram.main.interfaces.tokenizer import Tokenizer
from vgram import VGramTokenizer, SplitLevel


@lru_cache(maxsize=10)
def train_default_tokenizer(size: int = 1000, iters: int = 1, split_level: SplitLevel = SplitLevel.NONE) -> Tokenizer:
    data_dir = "/Users/aleksandr.khvorov/jb/grazie/grazie-datasets/data/"
    files = [data_dir + "stardust/all-texts.txt"]

    tokenizer = VGramTokenizer(size, split_level=split_level, verbose=True)
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
    tokenizer = VGramTokenizer(200, split_level=SplitLevel.LINE, verbose=True)
    tokenizer.fit(["hello", "hello world"] * 1000, iters=15)
    encoded_seq = tokenizer.encode("hello world")
    assert len(encoded_seq) == 1
    assert len(tokenizer.encode("hello")) == 1
    print(encoded_seq)
    decoded = tokenizer.decode(encoded_seq)
    print(decoded)
    assert decoded == "hello world"

    assert len(tokenizer.get_vocab()) == 10
    assert set(tokenizer.get_vocab()) == {'h', 'e', 'l', 'o', ' ', 'w', 'o', 'r', 'd', 'hello', 'hello world'}


def test_fit_twice():
    tokenizer = VGramTokenizer(200, split_level=SplitLevel.NONE, verbose=True)
    tokenizer.fit(["hello", "hello world"], iters=10000)
    assert len(tokenizer.encode("hello world")) == 1
    assert len(tokenizer.encode("hello")) == 1

    tokenizer.fit("pip install pyvgram", iters=10000)
    encoded_seq = tokenizer.encode("pip install pyvgram")
    assert len(encoded_seq) == 1

    print(encoded_seq)
    decoded = tokenizer.decode(encoded_seq)
    print(decoded)
    assert decoded == "pip install pyvgram"

    encoded_seq = tokenizer.encode("hello world")
    assert len(encoded_seq) > 1
    assert len(tokenizer.encode("hello")) > 1

    tokenizer.fit(["hello", "hello world", "pip install pyvgram"], iters=10000)
    assert len(tokenizer.encode("hello")) == 1
    assert len(tokenizer.encode("hello world")) == 1
    assert len(tokenizer.encode("pip install pyvgram")) == 1

    print(tokenizer.get_vocab())
    assert len(tokenizer.get_vocab()) == 21
    assert set(tokenizer.get_vocab()) == {'h', 'hello', 'hello world', 'e', 'l', 'o', ' ', 'w', 'r', 'd', 'p',
                                          'pip install pyvgram', 'i', 'n', 's', 't', 'a', 'y', 'v', 'g', 'm'}


def test_save_and_load():
    # tokenizer = train_default_tokenizer(200, 1)
    tokenizer = VGramTokenizer(200, split_level=SplitLevel.NONE, verbose=True)
    tokenizer.fit(["hello", "hello world"] * 1000, iters=15)
    path = ".tokenizer.tok"
    try:
        tokenizer.save_pretrained(path)
        loaded_tokenizer = VGramTokenizer.from_pretrained(path)
        assert tokenizer == loaded_tokenizer
    finally:
        if os.path.exists(path):
            os.remove(path)


def learn_big_dict(split_level: SplitLevel = SplitLevel.WORD, files_num: int = 1):
    data_dir = "/Users/aleksandr.khvorov/jb/grazie/grazie-datasets/data/"
    # files = [data_dir + "stardust/all-texts.txt"]
    files = [data_dir + f"openwebtext-parts-100/{i}.txt" for i in range(files_num)]

    size = 50000
    iters = 5
    tokenizer = VGramTokenizer(size, split_level, verbose=True)
    tokenizer.train(files, iters=iters)
    encoded_seq = tokenizer.encode("hello world")
    print(encoded_seq)
    decoded = tokenizer.decode(encoded_seq)
    assert decoded == "hello world"
    print([tokenizer.decode([i]) for i in tokenizer.encode("fix bug")])
    print(tokenizer.get_vocab()[:10])

    file_str = "".join(str(i) for i in range(files_num))
    tokenizer.save_pretrained(f"../saved/{size // 1000}k_owta_split_{split_level}_{file_str}_it{iters}.json")


def load_dict(path: str = f"../saved/20k_owta_0.json"):
    tokenizer = VGramTokenizer.from_pretrained(path)
    en = tokenizer.encode("hello")
    vocab = tokenizer.get_vocab()
    a = 0


if __name__ == '__main__':
    # test_fit()
    test_fit_twice()
    # test_train()
    # test_words_level()
    # test_save_and_load()
    # learn_big_dict(words_level=True)
    # load_dict(f"../saved/50k_owta_0_it5.json")
    # load_dict(f"../saved/20k_owta_nosplit_0_it3.json")
