#coding=utf-8
from __future__ import division

import numpy as np

import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

from cifar10_tf import train_data, test_data
from data_utils import IMG_SHAPE, LBL_COUNT


class Config(object):
    # hyper parameters for optimizer.
    init_learning_rate = 0.1
    weight_decay = 0.0005
    momentum = 0.9
    use_nesterov = True
    batch_size = 16

    test_batch_size = 1000

    # hyper parameters for model architecture.
    cardinality = 8  # how many split
    num_block = 3
    depth = 64  # out channel
    reduction_ratio = 4

    # the filter depths of the convolution blocks.
    out_dims = [64, 64, 128, 256]

    # hyper parameters for training task.
    total_epochs = 100


def Evaluate(sess, image_test, label_test, test_batch_size):
    test_acc = 0.0
    test_loss = 0.0

    for i, test_pre_index in range(test_iteration):
        test_batch_x = image_test[test_pre_index:test_pre_index + add]
        test_batch_y = label_test[test_pre_index:test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration  # average loss
    test_acc /= test_iteration  # average accuracy

    return test_acc, test_loss


class SE_ResNeXt(object):
    def __init__(self, x, num_block, depth, out_dims, cardinality,
                 reduction_ratio, is_training):
        self.is_training = is_training
        self.out_dims = out_dims
        self.cardinality = cardinality
        self.num_block = num_block
        self.depth = depth
        self.reduction_ratio = reduction_ratio

        self.model = self.build_SEnet(x)

    def batch_norm(self, x, is_training, scope):
        with arg_scope(
            [batch_norm],
                scope=scope,
                updates_collections=None,
                decay=0.9,
                center=True,
                scale=True,
                zero_debias_moving_mean=True):
            return tf.cond(
                is_training,
                lambda: batch_norm(inputs=x, is_training=is_training, reuse=None),
                lambda: batch_norm(inputs=x, is_training=is_training, reuse=True))

    def conv_bn_layer(self,
                      x,
                      filters,
                      filter_size,
                      stride,
                      scope,
                      padding="same"):
        with tf.name_scope(scope):
            x = tf.layers.conv2d(
                inputs=x,
                filters=filters,
                kernel_size=filter_size,
                strides=stride,
                padding=padding,
                use_bias=False)
            x = self.batch_norm(
                x, is_training=self.is_training, scope=scope + "_batch1")
            return tf.nn.relu(x)

    def transform_layer(self, x, stride, depth, scope):
        x = self.conv_bn_layer(
            x,
            filters=depth,
            filter_size=1,
            stride=1,
            padding="same",
            scope=scope + "_trans1")
        return self.conv_bn_layer(
            x,
            filters=depth,
            filter_size=3,
            stride=stride,
            padding="same",
            scope=scope + "_trans2")

    def split_layer(self, input_x, stride, depth, layer_name, cardinality):
        with tf.name_scope(layer_name):
            layer_splits = []
            for i in range(cardinality):
                layer_splits.append(
                    self.transform_layer(input_x, stride, depth,
                                         layer_name + "_splitN_" + str(i)))
            return tf.concat(layer_splits, axis=3)  # concatenate along channel

    def transition_layer(self, x, out_dim, scope):
        """A 1 x 1 convolution.
        """

        return self.conv_bn_layer(
            x,
            filters=out_dim,
            filter_size=1,
            stride=1,
            padding="same",
            scope=scope)

    def squeeze_excitation_layer(self, input_x, out_dim, reduction_ratio,
                                 layer_name):
        with tf.name_scope(layer_name):

            pool = global_avg_pool(input_x)
            squeeze = tf.layers.dense(
                pool,
                use_bias=False,
                units=out_dim / reduction_ratio,
            )
            squeeze = tf.nn.relu(squeeze)
            excitation = tf.layers.dense(
                squeeze, units=out_dim, use_bias=False)
            excitation = tf.nn.sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            return input_x * excitation

    def residual_layer(self, input_x, out_dim, layer_num, cardinality, depth,
                       reduction_ratio, num_block):
        for i in range(num_block):
            # The input here must follow channel last format
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1

            x = self.split_layer(
                input_x,
                stride=stride,
                cardinality=cardinality,
                depth=depth,
                layer_name="split_layer_" + layer_num + "_" + str(i))
            x = self.transition_layer(
                x,
                out_dim=out_dim,
                scope="trans_layer_" + layer_num + "_" + str(i))
            x = self.squeeze_excitation_layer(
                x,
                out_dim=out_dim,
                reduction_ratio=reduction_ratio,
                layer_name="squeeze_layer_" + layer_num + "_" + str(i))

            if flag is True:
                pad_input_x = tf.layers.average_pooling2d(
                    input_x, pool_size=[2, 2], strides=2, padding="same")
                pad_input_x = tf.pad(
                    pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]])
            else:
                pad_input_x = input_x
            input_x = tf.nn.relu(x + pad_input_x)
        return input_x

    def build_SEnet(self, input_x):
        input_x = self.conv_bn_layer(
            input_x,
            filters=self.out_dims[0],
            filter_size=3,
            stride=1,
            scope="first_layer")

        for i, out_dim in enumerate(self.out_dims[1:]):
            x = self.residual_layer(
                (x if i else input_x),
                out_dim=out_dim,
                num_block=self.num_block,
                depth=self.depth,
                cardinality=self.cardinality,
                reduction_ratio=self.reduction_ratio,
                layer_num=str(i + 1))

        x = global_avg_pool(x)
        x = flatten(x)
        return tf.layers.dense(inputs=x, use_bias=False, units=LBL_COUNT)


def train(conf):
    # Step 1: Load training and testing data
    image_train, label_train = train_data()
    image_test, label_test = train_data()
    total_train_sample = len(image_train)

    # Step 2: Define TF placeholders.
    images = tf.placeholder(
        tf.float32, shape=[None, IMG_SHAPE[1], IMG_SHAPE[2], IMG_SHAPE[0]])
    labels = tf.placeholder(tf.float32, shape=[None, LBL_COUNT])

    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    # Step 3: Build the network.
    logits = SE_ResNeXt(
        images,
        conf.num_block,
        conf.depth,
        conf.out_dims,
        conf.cardinality,
        conf.reduction_ratio,
        is_training=training_flag).model

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    # add the L2 regularization.
    l2_loss = tf.add_n(
        [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

    # Step 4: Define the optimizer.
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=conf.momentum,
        use_nesterov=conf.use_nesterov)
    train = optimizer.minimize(cost + l2_loss * conf.weight_decay)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Step 5: Begin training and testing tasks.
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        ckpt = tf.train.get_checkpoint_state("./model")
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        epoch_learning_rate = conf.init_learning_rate
        for epoch in range(1, conf.total_epochs + 1):
            if epoch % 30 == 0:
                epoch_learning_rate = epoch_learning_rate / 10

            train_acc = 0.0
            train_loss = 0.0
            for batch_id, start_index in enumerate(
                    range(0, total_train_sample, conf.batch_size)):
                end_index = min(start_index + conf.batch_size,
                                total_train_sample)
                batch_image = image_train[start_index:end_index]
                batch_label = label_train[start_index:end_index]

                train_feed_dict = {
                    images: batch_image,
                    labels: batch_label,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }

                _, batch_loss = sess.run(
                    [train, cost], feed_dict=train_feed_dict)
                batch_acc = accuracy.eval(feed_dict=train_feed_dict)

                train_loss += batch_loss
                train_acc += batch_acc
                print("Epoch %d, batch %d, loss = %.6f , acc = %.6f " %
                      (epoch, batch_id, batch_loss, batch_acc))

            train_loss /= iteration  # average loss
            train_acc /= iteration  # average accuracy

            test_acc, test_loss = Evaluate(sess)
            print(("epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, "
                   "test_loss: %.4f, test_acc: %.4f \n") %
                  (epoch, total_epochs, train_loss, train_acc, test_loss,
                   test_acc))
            saver.save(sess=sess, save_path="./model/ResNeXt.ckpt")


if __name__ == "__main__":
    conf = Config()
    train(conf)
