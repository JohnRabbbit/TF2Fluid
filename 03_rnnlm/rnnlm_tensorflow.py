#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from load_data_tf import ptb_raw_data, gen_data


class LMConfig(object):
    """Configuration of language model"""
    batch_size = 64
    max_sequence_length = 20
    stride = 3

    embedding_dim = 64
    hidden_dim = 128
    num_layers = 2

    learning_rate = 1e-3

    num_passes = 5


class PTBInput(object):
    def __init__(self, config, data):
        self.batch_size = config.batch_size
        self.max_sequence_length = config.max_sequence_length
        self.vocab_size = config.vocab_size

        self.input_data, self.targets = gen_data(
            data, self.batch_size, self.max_sequence_length, shuffle=True)

        self.batch_len = self.input_data.shape[0]
        self.cur_batch = 0

    def next_batch(self):
        x = self.input_data[self.cur_batch]
        y = self.targets[self.cur_batch]

        y_ = np.zeros((y.shape[0], self.vocab_size), dtype=np.bool)
        for i in range(y.shape[0]):
            y_[i][y[i]] = 1

        self.cur_batch = (self.cur_batch + 1) % self.batch_len
        return x, y_


class RNNLM(object):
    def __init__(self, config, is_training=True):
        self.max_sequence_length = config.max_sequence_length
        self.vocab_size = config.vocab_size

        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        self.learning_rate = config.learning_rate

        # build the model
        self.placeholders()
        self._logits, self._pred = self.rnn()
        self.cost = self.cost()
        self.optim = self.optimize()
        self.word_error = self.word_error()

    def placeholders(self):
        self._inputs = tf.placeholder(tf.int32,
                                      [None, self.max_sequence_length])
        self._targets = tf.placeholder(tf.int32, [None, self.vocab_size])

    def input_embedding(self):
        embedding = tf.get_variable(
            "embedding", [self.vocab_size, self.embedding_dim],
            dtype=tf.float32)
        return tf.nn.embedding_lookup(embedding, self._inputs)

    def rnn(self):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self.hidden_dim, state_is_tuple=True)

        cells = [lstm_cell() for _ in range(self.num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        _inputs = self.input_embedding()
        _outputs, _ = tf.nn.dynamic_rnn(
            cell=cell, inputs=_inputs, dtype=tf.float32)

        last = _outputs[:, -1, :]
        logits = tf.layers.dense(inputs=last, units=self.vocab_size)
        prediction = tf.nn.softmax(logits)

        return logits, prediction

    def cost(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self._logits, labels=self._targets)
        return tf.reduce_mean(cross_entropy)

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(self.cost)

    def word_error(self):
        mistakes = tf.not_equal(
            tf.argmax(self._targets, 1), tf.argmax(self._pred, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


def train():
    config = LMConfig()
    train_data, _, _, words, word_to_id = ptb_raw_data("data")
    config.vocab_size = len(words)

    input_train = PTBInput(config, train_data)
    batch_len = input_train.batch_len
    model = RNNLM(config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for pass_id in range(config.num_passes):
            for batch_id in range(batch_len):
                x_batch, y_batch = input_train.next_batch()

                feed_dict = {model._inputs: x_batch, model._targets: y_batch}
                sess.run([model.optim, model.cost], feed_dict=feed_dict)

                if batch_id and not batch_id % 5:
                    print("Pass %d, Batch %d, Loss %.4f" % (pass_id, batch_id,
                                                            cost))


if __name__ == "__main__":
    train()
