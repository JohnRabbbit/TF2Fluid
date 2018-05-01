#!/usr/bin/env python
#coding=utf-8
from collections import defaultdict

from tensorflow.python.client import device_lib


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


def get_available_gpus():
    """Returns a list of available GPU devices names.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]
