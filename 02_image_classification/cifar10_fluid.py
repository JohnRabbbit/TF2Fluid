import pdb
import os
import tarfile
import itertools
import cPickle
import numpy as np

from data_utils import download_data, color_preprocessing, IMG_SHAPE


def reader_creator(filename, sub_name, shuffle=True):
    def __get_data(batch):
        return batch["data"].astype(np.float32), batch["labels"]

    def reader():
        with tarfile.open(filename, mode="r") as fin:
            names = [
                each_item.name for each_item in fin
                if sub_name in each_item.name
            ]

            data, labels = __get_data(cPickle.load(fin.extractfile(names[0])))
            for data_file in names[1:]:
                data_batch, label_batch = __get_data(
                    cPickle.load(fin.extractfile(data_file)))
                data = np.append(data, data_batch, axis=0)
                labels = np.append(labels, label_batch, axis=0)

        data = data.reshape([-1] + IMG_SHAPE)
        data = color_preprocessing(data)

        if shuffle:
            indices = np.random.permutation(len(data))
            data = data[indices]
            labels = labels[indices]

        for sample, label in itertools.izip(data, labels):
            yield sample, int(label)

    return reader


def train_data(sub_name="data_batch", shuffle=True):
    return reader_creator(download_data(), sub_name, shuffle)


def test_data(sub_name="test_batch", shuffle=False):
    return reader_creator(download_data(), sub_name, shuffle)


if __name__ == "__main__":
    for data_batch in train_data()():
        print data_batch
        break

    for data_batch in test_data()():
        print data_batch
        break
