#!/usr/bin/env python
#coding=utf-8
from itertools import izip

__all__ = ["data_reader"]


def data_reader(src_file_name, trg_file_name, src_vocab_file, trg_vocab_file):
    def __load_dict(dict_file_path):
        word_dict = {}
        with open(dict_file_path, "r") as fdict:
            for idx, line in enumerate(fdict):
                if idx < 2: continue
                word_dict[line.strip().split("\t")[0]] = idx - 2

        return word_dict

    def __reader():
        src_dict = __load_dict(src_vocab_file)
        trg_dict = __load_dict(trg_vocab_file)

        with open(src_file_name, "r") as fsrc, open(trg_file_name,
                                                    "r") as ftrg:
            for src, trg in izip(fsrc, ftrg):
                src_words = src.strip().split()
                trg_words = trg.strip().split()

                src_ids = [src_dict[w] for w in src_words]
                trg_ids = [trg_dict[w] for w in trg_words]
                yield src_ids, trg_ids

    return __reader


if __name__ == "__main__":
    src_file_name = "data/train_src.txt"
    src_vocab_file = "data/train_src.vocab"

    trg_file_name = "data/train_trg.txt"
    trg_vocab_file = "data/train_trg.vocab"

    for idx, data in enumerate(
            data_reader(src_file_name, trg_file_name, src_vocab_file,
                        trg_vocab_file)()):
        # if idx > 5: break
        print data
