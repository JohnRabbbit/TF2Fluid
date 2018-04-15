#!/usr/bin/env python
#coding=utf-8

import os

from data_utils import read_words, build_vocab


def train_data(data_dir="data"):
    data_path = os.path.join(data_dir, "ptb.train.txt")
    _, word_to_id = build_vocab(data_path)

    with open(data_path, "r") as ftrain:
        for line in ftrain:
            words = line.strip().split()
            word_ids = [word_to_id[w] for w in words]
            yield word_ids[0:-1], word_ids[1:]
