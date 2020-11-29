from main.interfaces.vgram_tokenizer import VGramTokenizer


def test_words_level():
    data_dir = "/Users/aleksandr.khvorov/jb/grazie/grazie-datasets/data/"
    files = [data_dir + "stardust/all-texts.txt"]

    tokenizer = VGramTokenizer(10000, words_level=True)
    tokenizer.train(files, iters=1, verbose=1)
    encoded_seq = tokenizer.encode("hello world")
    print(encoded_seq)
    decoded = tokenizer.decode(encoded_seq)
    assert decoded == "hello world"
    print([tokenizer.decode([i]) for i in tokenizer.encode("fix bug")])
    print([tokenizer.coder.decode(i) for i in tokenizer.dict.seqs][:10])


def test_1():
    data_dir = "/Users/aleksandr.khvorov/jb/grazie/grazie-datasets/data/"
    files = [data_dir + "stardust/all-texts.txt"]

    tokenizer = VGramTokenizer(10000, words_level=False)
    tokenizer.train(files, iters=5, verbose=1)
    encoded_seq = tokenizer.encode("hello world")
    print(encoded_seq)
    decoded = tokenizer.decode(encoded_seq)
    assert decoded == "hello world"
    print([tokenizer.decode([i]) for i in tokenizer.encode("fix bug")])


def test_fit():
    tokenizer = VGramTokenizer(200)
    tokenizer.fit(["hello", "hello world"] * 1000, iters=15, verbose=1)
    encoded_seq = tokenizer.encode("hello world")
    assert len(encoded_seq) == 1
    print(encoded_seq)
    decoded = tokenizer.decode(encoded_seq)
    print(decoded)
    assert decoded == "hello world"


if __name__ == '__main__':
    # test_fit()
    # test_1()
    test_words_level()
