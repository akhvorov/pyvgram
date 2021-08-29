from vgram.main.algorithm.int_dictionary import IntDictionary


def test_search():
    seqs = [
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2, 0),
    ]
    dict = IntDictionary(seqs)
    seq = dict.search((1, 2, 1, 0))
    assert seq == 3


def test_linear_parse():
    seqs = [
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2, 0),
    ]
    dict = IntDictionary(seqs)
    ids = dict.linear_parse([1, 2, 1, 0])
    assert ids == [3, 6, 4]


def test_weighted_parse():
    seqs = [
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2, 0),
    ]
    dict = IntDictionary(seqs)
    freqs = [1, 1, 2, 1, 3, 3, 1]
    ids = dict.weighted_parse([1, 2, 1, 0], freqs, total_freq=sum(freqs))
    assert ids == [3, 6, 4]

    freqs = [10, 1, 2, 10, 1, 3, 1]
    ids = dict.weighted_parse([1, 2, 1, 0], freqs, total_freq=sum(freqs))
    assert ids == [3, 6, 3, 0]


if __name__ == '__main__':
    # test_search()
    # test_linear_parse()
    test_weighted_parse()
