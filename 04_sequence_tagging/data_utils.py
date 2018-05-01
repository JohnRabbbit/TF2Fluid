#!/usr/bin/env python
#coding=utf-8


def load_tag_dict(tag_dict_file):
    tag_dict = {}
    with open(tag_dict_file, "r") as fin:
        for idx, line in enumerate(fin):
            tag_dict[str(idx)] = line.strip().split("\t")[0]
    return tag_dict
