from main.interfaces.vgram_tokenizer import VGramTokenizer


def test_1():
    data_dir = "/Users/aleksandr.khvorov/jb/grazie/grazie-datasets/data/"
    files = [data_dir + "results_with_commits_hash_common_delete_duplicates.txt"]

    tokenizer = VGramTokenizer(200)
    tokenizer.train(files, iters=1, verbose=1)
    encoded_seq = tokenizer.encode("hello world")
    print(encoded_seq)
    decoded = tokenizer.decode(encoded_seq)
    print(decoded)


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
    test_fit()
