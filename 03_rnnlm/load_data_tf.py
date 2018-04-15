#!/usr/bin/python
# -*- coding: utf-8 -*-

__all__ = [
    "ptb_raw_data",
    "gen_data",
    "to_words",
]

import os
import numpy as np

from data_utils import read_words, build_vocab


def to_words(sentence, words):
    return list(map(lambda x: words[x], sentence))


def ptb_raw_data(data_path=None):
    """load the original PTB dataset.
    """

    def _file_to_word_ids(filename, word_to_id):
        """
        Convert word string to word index according to the given word
        dictionary. Each line in the data file is an example. words are
        separated by blanks.
        """
        data = read_words(filename)
        return [word_to_id[x] for x in data if x in word_to_id]

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    words, word_to_id = build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)

    return train_data, valid_data, test_data, words, word_to_id


def gen_data(raw_data, batch_size, num_steps=20, stride=3, shuffle=False):
    data_len = len(raw_data)


    sentences = []
    next_words = []
    for i in range(0, data_len - num_steps, stride):
        sentences.append(raw_data[i:(i + num_steps)])
        next_words.append(raw_data[i + num_steps])

    sentences = np.array(sentences)
    next_words = np.array(next_words)

    batch_count = len(sentences) // batch_size
    x = np.reshape(sentences[:(batch_count * batch_size)],
                   [batch_count, batch_size, -1])
    y = np.reshape(next_words[:(batch_count * batch_size)],
                   [batch_count, batch_size])
    return x, y


def valid_data():
    train_data, valid_data, test_data, words, word_to_id = ptb_raw_data("data")
    x_train, y_train = gen_data_tf(train_data, batch_size=5, shuffle=True)

    print(x_train.shape)
    print(to_words(x_train[100, 3], words))
    print(words[np.argmax(y_train[100, 3])])


if __name__ == "__main__":
    valid_data()
