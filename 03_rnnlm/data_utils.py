#!/usr/bin/env python
#coding=utf-8

import codecs
from collections import Counter

__all__ = [
    "read_words",
    "build_vocab",
]


def read_words(filename):
    """Read word strings from the given file.
    """
    with codecs.open(filename, "r", encoding="utf-8") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    """Biuld the word vocabulary.
    """

    data = read_words(filename)
    counter = Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id
