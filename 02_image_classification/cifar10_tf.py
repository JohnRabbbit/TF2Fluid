import os
import sys
import cPickle
import random
import tarfile
import numpy as np

from data_utils import color_preprocessing, download_data, IMG_SHAPE, LBL_COUNT

__all__ = [
    "train_data",
    "test_data",
]


def __prepare_data(sub_name, shuffle=True):
    def __get_data(batch):
        return batch["data"], batch["labels"]

    filename = download_data()

    with tarfile.open(filename, mode="r") as fin:
        names = [
            each_item.name for each_item in fin if sub_name in each_item.name
        ]

        data, labels = __get_data(cPickle.load(fin.extractfile(names[0])))
        for data_file in names[1:]:
            data_batch, label_batch = __get_data(
                cPickle.load(fin.extractfile(data_file)))
            data = np.append(data, data_batch, axis=0)
            labels = np.append(labels, label_batch, axis=0)

    # generate dense labels
    labels = np.array(
        [[float(i == label) for i in range(LBL_COUNT)] for label in labels])
    data = data.reshape([-1] + IMG_SHAPE)
    data = color_preprocessing(data)

    # tranpose the original order: [N, C, H, W] into [N, H, W, C]
    data = data.transpose([0, 2, 3, 1])

    if shuffle:
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]

    return data, labels


def test_data():
    return __prepare_data("test_batch", False)


def train_data():
    return __prepare_data("data_batch")


if __name__ == "__main__":
    train_data, train_labels = train_data()
    test_data, test_labels = test_data()
