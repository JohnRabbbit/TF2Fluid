#!/usr/bin/env python
#coding=utf-8
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_gpus():
    """Returns a list of available GPU devices names.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


class RNNLM(object):
    def __init__(self, config, curwd, nxtwd, seq_len, is_training=True):
        self.batch_size = tf.size(nxtwd)

        self.time_major = config.time_major
        self.vocab_size = config.vocab_size

        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        self.learning_rate = config.learning_rate

        # build the model
        self.cost = self.make_parallel(
            self.build_model,
            len(get_available_gpus()),
            curwd=curwd,
            nxtwd=nxtwd,
            seq_len=seq_len)
        self.optim = self.optimize()

    def build_model(self, curwd, nxtwd, seq_len):
        embedding = self.input_embedding(curwd)
        logits, prediction = self.rnn(embedding, seq_len)
        return self.cost(seq_len, nxtwd, logits)

    def make_parallel(self, fn, num_gpus, **kwargs):
        in_splits = {}
        for k, v in kwargs.items():
            in_splits[k] = tf.split(v, num_gpus)

        out_split = []
        for i in range(num_gpus):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
                with tf.variable_scope(
                        tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    out_i = fn(**{k: v[i] for k, v in in_splits.items()})
                    out_split.append(out_i)

        return tf.reduce_sum(tf.add_n(out_split)) / tf.to_float(
            self.batch_size)

    def input_embedding(self, curwd):
        embedding = tf.get_variable(
            "embedding", [self.vocab_size, self.embedding_dim],
            dtype=tf.float32)
        return tf.nn.embedding_lookup(embedding, curwd)

    def rnn(self, embedding, seq_len):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self.hidden_dim, state_is_tuple=True)

        cells = [lstm_cell() for _ in range(self.num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        _outputs, _ = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=embedding,
            dtype=tf.float32,
            sequence_length=seq_len,
            time_major=self.time_major,
            swap_memory=True)

        logits = tf.layers.dense(
            inputs=_outputs, units=self.vocab_size, use_bias=False)

        return logits, tf.nn.softmax(logits)

    def cost(self, seq_len, nxtwd, logits):
        def __get_max_time(tensor):
            time_axis = 0 if self.time_major else 1
            return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=nxtwd, logits=logits)
        target_weights = tf.sequence_mask(
            seq_len, __get_max_time(logits), dtype=logits.dtype)

        if self.time_major:
            target_weights = tf.transpose(target_weights)

        return cross_entropy * target_weights

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(self.cost, colocate_gradients_with_ops=True)

    def word_error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.nxtwd, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
