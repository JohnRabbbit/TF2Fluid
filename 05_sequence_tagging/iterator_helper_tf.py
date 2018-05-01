#!/usr/bin/env python
#coding=utf-8

import os
import sys
import collections

import tensorflow as tf

__all__ = [
    "get_data_iterator",
    "file_content_iterator",
]


class BatchedInput(
        collections.namedtuple(
            "BatchedInput",
            ("initializer", "source", "target", "sequence_length"))):
    pass


def get_data_iterator(src_file_name,
                      trg_file_name,
                      src_vocab_file,
                      trg_vocab_file,
                      batch_size,
                      pad_token="</p>",
                      max_sequence_length=None,
                      unk_id=1,
                      num_parallel_calls=4,
                      num_buckets=5,
                      output_buffer_size=102400,
                      is_training=True):
    def __get_word_dict(vocab_file_path, unk_id):
        return tf.contrib.lookup.index_table_from_file(
            vocabulary_file=vocab_file_path,
            key_column_index=0,
            default_value=unk_id)

    src_dataset = tf.data.TextLineDataset(src_file_name)
    trg_dataset = tf.data.TextLineDataset(trg_file_name)

    dataset = tf.data.Dataset.zip((src_dataset, trg_dataset))
    if is_training:
        dataset = dataset.shuffle(
            buffer_size=output_buffer_size, reshuffle_each_iteration=True)

    src_trg_dataset = dataset.map(
        lambda src, trg: (tf.string_split([src]).values, \
                tf.string_split([trg]).values),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    src_dict = __get_word_dict(src_vocab_file, unk_id)
    trg_dict = __get_word_dict(trg_vocab_file, unk_id)

    src_pad_id = tf.cast(src_dict.lookup(tf.constant(pad_token)), tf.int32)
    trg_pad_id = tf.cast(trg_dict.lookup(tf.constant(pad_token)), tf.int32)

    # convert word string to word index
    src_trg_dataset = src_trg_dataset.map(
        lambda src, trg: (
                tf.cast(src_dict.lookup(src), tf.int32),
                tf.cast(trg_dict.lookup(trg), tf.int32)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Add in sequence lengths.
    src_trg_dataset = src_trg_dataset.map(
        lambda src, trg: (src, trg, tf.size(src)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    def __batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # trg
                tf.TensorShape([]),  #seq_len
            ),
            padding_values=(src_pad_id, trg_pad_id, 0, ))

    if num_buckets > 1:

        def __key_func(unused_1, unused_2, seq_len):
            if max_sequence_length:
                bucket_width = (
                    max_sequence_length + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            bucket_id = seq_len // bucket_width,
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def __reduce_func(unused_key, windowed_data):
            return __batching_func(windowed_data)

        batched_dataset = src_trg_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=__key_func,
                reduce_func=__reduce_func,
                window_size=batch_size))

    else:
        batched_dataset = __batching_func(curwd_nxtwd_dataset)

    batched_iter = batched_dataset.make_initializable_iterator()
    src_ids, trg_ids, seq_len = batched_iter.get_next()
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target=trg_ids,
        sequence_length=seq_len)


def file_content_iterator(file_name):
    with open(file_name, "r") as f:
        for line in f.readlines():
            yield line.strip()


if __name__ == "__main__":
    src_file_name = "data/train_src.txt"
    src_vocab_file = "data/train_src.vocab"

    trg_file_name = "data/train_trg.txt"
    trg_vocab_file = "data/train_trg.vocab"

    batch_size = 2

    iterator = get_train_iterator(src_file_name, trg_file_name, src_vocab_file,
                                  trg_vocab_file, batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)

        for i in range(5):
            ids1, ids2, ids_len = sess.run(
                [iterator.source, iterator.target, iterator.sequence_length])
            print ids_len
