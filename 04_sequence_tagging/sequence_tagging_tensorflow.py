# -*- coding: utf-8 -*
import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper

from iterator_helper_tf import get_data_iterator, file_content_iterator
from config import ModelConfig
from data_utils import load_tag_dict


class NER_net(object):
    def __init__(self, iterator, config, scope_name="ner_net"):
        self.iterator = iterator
        self.config = config

        with tf.variable_scope(scope_name) as scope:
            self.__build_net()

    def __build_net(self):
        source = self.iterator.source
        target = self.iterator.target
        seq_len = self.iterator.sequence_length
        config = self.config

        embedding = tf.get_variable(
            "embedding", [conf.src_vocab_size, config.embedding_dim],
            dtype=tf.float32)
        self.x = tf.nn.embedding_lookup(embedding, source)
        self.y = target

        cell_forward = tf.contrib.rnn.BasicLSTMCell(conf.unit_num)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(conf.unit_num)
        if config.drop_rate is not None:
            cell_forward = DropoutWrapper(
                cell_forward,
                input_keep_prob=1.0,
                output_keep_prob=config.drop_rate)
            cell_backward = DropoutWrapper(
                cell_backward,
                input_keep_prob=1.0,
                output_keep_prob=config.drop_rate)

        outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            cell_forward,
            cell_backward,
            self.x,
            dtype=tf.float32,
            sequence_length=seq_len)

        forward_out, backward_out = outputs
        outputs = tf.concat([forward_out, backward_out], axis=2)

        W = tf.get_variable("projection_w",
                            [2 * config.unit_num, config.tag_num])
        b = tf.get_variable("projection_b", [config.tag_num])
        x_reshape = tf.reshape(outputs, [-1, 2 * config.unit_num])
        projection = tf.matmul(x_reshape, W) + b

        self.outputs = tf.reshape(projection,
                                  [config.batch_size, -1, config.tag_num])

        max_sequence_in_batch = tf.to_int32(
            tf.reduce_max(self.iterator.sequence_length))
        self.seq_length = tf.convert_to_tensor(
            config.batch_size * [max_sequence_in_batch], dtype=tf.int32)
        self.log_likelihood, self.transition_params = (
            tf.contrib.crf.crf_log_likelihood(self.outputs, self.y,
                                              self.seq_length))

        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


def train(net, iterator, config, sess):
    saver = tf.train.Saver()

    current_batch = 0
    current_epoch = 0
    while True:
        try:
            tf_unary_scores, tf_transition_params, _, losses = sess.run(
                [net.outputs, net.transition_params, net.train_op, net.loss])

            current_batch += 1
            if current_batch and current_epoch % 10 == 0:
                print("Epoch %d, Batch %d Loss %.5f" % (current_epoch,
                                                        current_batch, losses))
        except tf.errors.OutOfRangeError:
            saver.save(sess,
                       os.path.join(os.getcwd(), config.tf_model_path,
                                    "model.ckpt"))

            current_epoch += 1
            if current_epoch > config.epoch_num: break
            sess.run(iterator.initializer)
            current_batch = 0
        except tf.errors.InvalidArgumentError:
            sess.run(iterator.initializer)
    print "training finished!"


def predict(conf, net, tag_dict, sess):
    saver = tf.train.import_meta_graph(
        os.path.join(conf.tf_model_path, "model.ckpt.meta"))
    saver.restore(sess, tf.train.latest_checkpoint(conf.tf_model_path))

    while True:
        try:
            tf_unary_scores, tf_transition_params = sess.run(
                [net.outputs, net.transition_params])
        except tf.errors.OutOfRangeError:
            break

        tf_unary_scores = np.squeeze(tf_unary_scores)

        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
            tf_unary_scores, tf_transition_params)

        tags = [tag_dict[str(tag)] for tag in viterbi_sequence]
        print tags


if __name__ == "__main__":
    conf = ModelConfig()

    if conf.job == "train":
        iterator = get_data_iterator(
            conf.train_src_file_name, conf.train_trg_file_name,
            conf.src_vocab_file, conf.trg_vocab_file, conf.batch_size)
    elif conf.job == "predict":
        iterator = get_data_iterator(
            conf.dev_src_file_name,
            conf.dev_trg_file_name,
            conf.src_vocab_file,
            conf.trg_vocab_file,
            1,
            is_training=False)
    else:
        print "Only support train and predict jobs."
        exit(0)

    net = NER_net(iterator, conf)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)

        if conf.job == "train":
            train(net, iterator, conf, sess)
        elif conf.job == "predict":
            predict(conf, net, load_tag_dict(conf.trg_vocab_file), sess)
