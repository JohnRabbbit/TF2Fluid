#coding=utf-8
import numpy as np
import tensorflow as tf

from tf_load_MNIST import load_MNIST


def data_iterator(dataset="training", path="data", batch_size=128):
    batch_idx = 0
    lbl, img = load_MNIST(dataset, path)
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(lbl))
        np.random.shuffle(idxs)
        shuf_features = img[idxs]
        shuf_labels = lbl[idxs]
        for batch_idx in range(0, len(lbl), batch_size):
            images_batch = shuf_features[batch_idx:
                                         batch_idx + batch_size] / 255.
            images_batch = images_batch.astype("float32")
            labels_batch = shuf_labels[batch_idx:
                                       batch_idx + batch_size].astype("int32")
            yield images_batch, labels_batch


def main():
    # define the network topology.
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(
        tf.int32, shape=[
            None,
        ])

    y = tf.layers.dense(inputs=x, units=10)
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    train_op = tf.train.AdamOptimizer().minimize(cross_entropy)

    # define the initializer.
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    train_reader = data_iterator()
    for step in range(100):
        images_batch, labels_batch = next(train_reader)
        _, loss_val = sess.run(
            [train_op, cross_entropy],
            feed_dict={
                x: images_batch,
                y_: labels_batch.astype("int32")
            })
        if step % 200 == 0:
            print("Cur Cost : %f" % loss_val)


if __name__ == "__main__":
    main()
