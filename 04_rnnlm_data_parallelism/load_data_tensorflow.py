#!/usr/bin/env python
#coding=utf-8
import os
import sys

import tensorflow as tf

from utils import build_dict


def get_dataset(file_name,
                vocab_file_path,
                batch_size,
                max_sequence_length=None,
                unk_id=1,
                bos_token="<bos>",
                eos_token="<eos>",
                num_parallel_calls=4,
                num_buckets=1,
                output_buffer_size=102400):
    dataset = tf.data.TextLineDataset(file_name)

    dataset = dataset.shuffle(
        buffer_size=output_buffer_size, reshuffle_each_iteration=True)

    curwd_nxtwd_dataset = tf.data.Dataset.zip((dataset, dataset))
    curwd_nxtwd_dataset = curwd_nxtwd_dataset.map(
        lambda curwd, nxtwd: (
                tf.string_split([curwd]).values,
                tf.string_split([nxtwd]).values),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    word_dict = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=vocab_file_path,
        key_column_index=0,
        default_value=unk_id)
    bos_id = tf.cast(word_dict.lookup(tf.constant(bos_token)), tf.int32)
    eos_id = tf.cast(word_dict.lookup(tf.constant(eos_token)), tf.int32)

    # convert word string to word index
    curwd_nxtwd_dataset = curwd_nxtwd_dataset.map(
        lambda curwd, nxtwd: (
                tf.cast(word_dict.lookup(curwd), tf.int32),
                tf.cast(word_dict.lookup(nxtwd), tf.int32)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    curwd_nxtwd_dataset = curwd_nxtwd_dataset.map(
        lambda curwd, nxtwd: (tf.concat(([bos_id], curwd), 0),
                          tf.concat((curwd, [eos_id]), 0)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Add in sequence lengths.
    curwd_nxtwd_dataset = curwd_nxtwd_dataset.map(
        lambda curwd, nxtwd: (curwd, nxtwd, tf.size(curwd), tf.size(nxtwd)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    def __batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([None]),  # current word
                tf.TensorShape([None]),  # next word
                tf.TensorShape([]),  # current_word_len
                tf.TensorShape([])  # next_word_len
            ),
            padding_values=(
                eos_id,  # current word
                eos_id,  # next word
                0,  # current_word_len, unused
                0  # next_word_len, unused
            ))

    if num_buckets > 1:

        def __key_func(unused_1, unused_2, curwd_len, nxtwd_len):
            if max_sequence_length:
                bucket_width = (
                    max_sequence_length + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            bucket_id = tf.maximum(curwd_len // bucket_width,
                                   nxtwd_len // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def __reduce_func(unused_key, windowed_data):
            return __batching_func(windowed_data)

        batched_dataset = curwd_nxtwd_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=__key_func,
                reduce_func=__reduce_func,
                window_size=batch_size))

    else:
        batched_dataset = __batching_func(curwd_nxtwd_dataset)

    batched_iter = batched_dataset.make_initializable_iterator()
    curwd, nxtwd, curwd_len, nxtwd_len = batched_iter.get_next()
    return batched_iter.initializer, curwd, nxtwd, curwd_len


if __name__ == "__main__":
    file_name = "data/ptb.train.txt"
    vocab_file_path = "data/vocab.txt"
    if not os.path.exists(vocab_file_path):
        build_dict(file_name, vocab_file_path)

    batch_size = 2

    initializer, curwd, nxtwd, seq_len = get_dataset(
        file_name, vocab_file_path, batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(initializer)

        for i in range(5):
            ids1, ids2, ids2_len = sess.run([curwd, nxtwd, seq_len])
            print ids2_len
