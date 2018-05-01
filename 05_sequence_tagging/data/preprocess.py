#!/usr/bin/env python
#coding=utf-8
import os
import sys

from collections import defaultdict

__all__ = [
    "preprocess_raw_data",
    "build_vocab",
]


def get_prefix(data_file_path):
    return os.path.splitext(data_file_path)[0]


def preprocess_raw_data(data_file_path):
    def __to_BIO_lbl(raw_lbl):
        bio_lbl = []

        prev_lbl = None
        for cur_lbl in raw_lbl:
            if cur_lbl == "O":
                bio_lbl.append(cur_lbl)
            else:
                prefix = "I-" if cur_lbl == prev_lbl else "B-"
                bio_lbl.append(prefix + cur_lbl)
            prev_lbl = cur_lbl
        return bio_lbl

    filename = get_prefix(data_file_path)
    with open(data_file_path,
              "r") as fin, open(filename + "_src.txt", "w") as fsrc, open(
                  filename + "_trg.txt", "w") as ftrg:
        src = []
        trg = []
        for line in fin:
            line = line.strip()
            if "DOCSTART" in line: continue

            if line:
                txt, lbl = line.strip().split("\t")
                src.append(txt)
                trg.append(lbl)
            else:
                if src and trg:
                    trg = __to_BIO_lbl(trg)

                    fsrc.write("%s\n" % (" ".join(src)))
                    ftrg.write("%s\n" % (" ".join(trg)))
                src = []
                trg = []

        if src and trg:
            fsrc.write("%s\n" % (" ".join(src)))
            ftrg.write("%s\n" % (" ".join(trg)))
            src = []
            trg = []


def build_vocab(data_file_path):
    filename = get_prefix(data_file_path)
    word_dict = defaultdict(int)

    with open(data_file_path, "r") as fin:
        for line in fin:
            words = line.strip().split()
            for w in words:
                word_dict[w] += 1

    all_words = sorted(word_dict.iteritems(), key=lambda x: x[1], reverse=True)

    with open(get_prefix(data_file_path) + ".vocab", "w") as fdict:
        fdict.write("</p>\t-1\n<unk>\t-1\n")
        for word, fre in all_words:
            fdict.write("%s\t%d\n" % (word, fre))


if __name__ == "__main__":
    for data_file_name in ["train", "dev"]:
        preprocess_raw_data(data_file_name)
        build_vocab(data_file_name + "_src.txt")
        build_vocab(data_file_name + "_trg.txt")
