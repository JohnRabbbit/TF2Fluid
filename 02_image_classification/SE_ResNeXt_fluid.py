import os

import paddle.v2 as paddle
import paddle.fluid as fluid

from cifar10_fluid import train_data, test_data
from data_utils import IMG_SHAPE, LBL_COUNT


class Config(object):
    # hyper parameters for optimizer.
    learning_rate = 0.1
    weight_decay = 0.0005
    momentum = 0.9
    use_nesterov = True
    batch_size = 16
    init_model = None

    test_batch_size = 1000

    # hyper parameters for model architecture.
    cardinality = 8  # how many split
    blocks = 3
    depth = 64  # out channel
    reduction_ratio = 4
    out_dims = [64, 128, 256]

    # hyper parameters for training task.
    total_epochs = 100


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

    def conv_bn_layer(self,
                      x,
                      num_filters,
                      filter_size,
                      stride,
                      padding,
                      groups=1,
                      act=None,
                      bias_attr=False):
        conv = fluid.layers.conv2d(
            input=x,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=act,
            bias_attr=bias_attr)
        return fluid.layers.batch_norm(input=conv, act=act)

    def transform_layer(self, x, stride, depth):
        x = self.conv_bn_layer(
            x,
            num_filters=depth,
            filter_size=1,
            stride=1,
            padding=0,
            act="relu")

        filter_size = 3
        return self.conv_bn_layer(
            x,
            num_filters=depth,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) / 2,
            act="relu")

    def split_layer(self, input_x, stride, depth, cardinality):
        layer_splits = []
        for i in range(cardinality):
            layer_splits.append(self.transform_layer(input_x, stride, depth))
        return fluid.layers.concat(
            layer_splits, axis=1)  # concatenate along channel

    def transition_layer(self, x, out_dim):
        return self.conv_bn_layer(
            x,
            num_filters=out_dim,
            filter_size=1,
            stride=1,
            padding=0,
            act="relu")

    def squeeze_excitation_layer(self, x, out_dim, reduction_ratio):
        """The Squeeze-and-Excitation blocks.
        """

        pool = fluid.layers.pool2d(
            input=x, pool_size=0, pool_type="avg", global_pooling=True)
        squeeze = fluid.layers.fc(
            input=pool, size=out_dim / reduction_ratio, act="relu")
        excitation = fluid.layers.fc(
            input=squeeze, size=out_dim, act="sigmoid")
        return fluid.layers.elementwise_mul(x=x, y=excitation, axis=0)

    def residual_layer(self, input_x, out_dim, cardinality, depth,
                       reduction_ratio, num_block):
        for i in range(num_block):
            input_dim = int(input_x.shape[1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1

            x = self.split_layer(
                input_x, stride=stride, cardinality=cardinality, depth=depth)
            x = self.transition_layer(x, out_dim=out_dim)
            x = self.squeeze_excitation_layer(
                x, out_dim=out_dim, reduction_ratio=reduction_ratio)

            if flag is True:
                pad_input_x = fluid.layers.pool2d(
                    input=input_x,
                    pool_stride=2,
                    pool_size=2,
                    pool_padding=0,
                    pool_type="avg")
                pad_input_x = fluid.layers.pad(
                    pad_input_x,
                    paddings=[
                        0,
                        0,  # padding settings for dimension 1
                        channel,
                        channel,  # padding settings for dimension 2
                        0,
                        0,  # padding settings for dimension 3
                        0,
                        0  # padding settings for dimension 4
                    ])
            else:
                pad_input_x = input_x

            input_x = fluid.layers.relu(x + pad_input_x)
        return input_x

    def build_SEnet(self, input_x):
        filter_size = 3
        input_x = self.conv_bn_layer(
            input_x,
            num_filters=64,
            filter_size=filter_size,
            stride=1,
            padding=(filter_size - 1) / 2,
            act="relu")

        for i, out_dim in enumerate(self.out_dims):
            x = self.residual_layer(
                (x if i else input_x),
                out_dim=out_dim,
                num_block=self.num_block,
                depth=self.depth,
                cardinality=self.cardinality,
                reduction_ratio=self.reduction_ratio)

        pool = fluid.layers.pool2d(
            input=x, pool_size=0, pool_type="avg", global_pooling=True)
        x = fluid.layers.fc(input=pool, size=LBL_COUNT, act="softmax")
        return x


def train(conf):
    images = fluid.layers.data(name="image", shape=IMG_SHAPE, dtype="float32")
    labels = fluid.layers.data(name="label", shape=[1], dtype="int64")

    out = SE_ResNeXt(
        images,
        conf.blocks,
        conf.depth,
        conf.out_dims,
        conf.cardinality,
        conf.reduction_ratio,
        is_training=True).model

    cost = fluid.layers.cross_entropy(input=out, label=labels)
    avg_cost = fluid.layers.mean(x=cost)
    accuracy = fluid.layers.accuracy(input=out, label=labels)

    optimizer = fluid.optimizer.Momentum(
        learning_rate=conf.learning_rate,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    opts = optimizer.minimize(avg_cost)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program(
            [avg_cost, accuracy])

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if conf.init_model is not None:
        fluid.io.load_persistables(exe, init_model)

    train_reader = paddle.batch(train_data(), batch_size=conf.batch_size)
    test_reader = paddle.batch(test_data(), batch_size=conf.batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[images, labels])

    for pass_id in range(conf.total_epochs):
        for batch_id, data in enumerate(train_reader()):
            loss = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost])
            print("Pass {0}, batch {1}, loss {2}".format(
                pass_id, batch_id, float(loss[0])))

        total_loss = 0.0
        total_acc = 0.0
        total_batch = 0
        for data in test_reader():
            loss, acc = exe.run(
                inference_program,
                feed=feeder.feed(data),
                fetch_list=[avg_cost, accuracy])
            total_loss += float(loss)
            total_acc += float(acc)
            total_batch += 1
        print("End pass {0}, test_loss {1}, test_acc {2}".format(
            pass_id, total_loss / total_batch, total_acc / total_batch))

        model_path = os.path.join(model_save_dir, str(pass_id))
        fluid.io.save_inference_model(model_path, ["image"], [out], exe)


if __name__ == "__main__":
    conf = Config()
    train(conf)
