#!/usr/bin/env python
#coding=utf-8

import codecs
from collections import Counter
from collections import defaultdict

__all__ = [
    "read_words",
    "build_vocab",
    "get_available_gpus",
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


def build_dict(file_name, save_path):
    word_dict = defaultdict(int)
    with open(file_name, "r") as fin:
        for line in fin:
            words = ["<bos>"] + line.strip().split() + ["<eos>"]
            for w in words:
                word_dict[w] += 1

    sorted_dict = sorted(
        word_dict.iteritems(), key=lambda x: x[1], reverse=True)

    with open(save_path, "w") as fdict:
        for w, freq in sorted_dict:
            fdict.write("%s\t%d\n" % (w, freq))
