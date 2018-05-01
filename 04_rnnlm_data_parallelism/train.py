#!/usr/bin/env python
#coding=utf-8

import tensorflow as tf

from rnnlm import LMConfig, RNNLM
from dataset_api_example import get_dataset


def train():
    model_config = LMConfig()
    initializer, curwd, nxtwd, nxtwd_len = get_dataset(
        model_config.train_data_path, model_config.vocab_file_path,
        model_config.batch_size)

    model = RNNLM(model_config, curwd, nxtwd, nxtwd_len)

    config = tf.ConfigProto()
    config.log_device_placement = True
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = False

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(initializer)

        pass_id = 0
        batch_id = 0

        while True:
            try:
                cost, _ = sess.run([model.cost, model.optim])

                batch_id += 1
                print("Pass %d, Batch %d, Loss %.4f" % (pass_id, batch_id,
                                                        cost))
            except tf.errors.OutOfRangeError:
                pass_id += 1
                if pass_id == model_config.num_passes: break
                sess.run(initializer)
                batch_id = 0
                continue


if __name__ == "__main__":
    train()
