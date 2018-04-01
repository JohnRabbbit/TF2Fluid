import numpy as np

import paddle.fluid as fluid
import paddle.v2 as paddle


def main():
    BATCH_SIZE = 128

    # define the input layers for the network.
    x = fluid.layers.data(name="img", shape=[1, 28, 28], dtype="float32")
    y_ = fluid.layers.data(name="label", shape=[1], dtype="int64")

    # define the network topology.
    y = fluid.layers.fc(input=x, size=10, act="softmax")
    loss = fluid.layers.cross_entropy(input=y, label=y_)
    avg_loss = fluid.layers.mean(loss)

    optimizer = fluid.optimizer.Adam(learning_rate=5e-3)
    optimizer.minimize(avg_loss)

    # define the executor.
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=5000),
        batch_size=BATCH_SIZE)
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y_])
    for pass_id in range(100):
        for batch_id, data in enumerate(train_reader()):
            loss = exe.run(
                fluid.framework.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_loss])
            print("Cur Cost : %f" % (np.array(loss[0])[0]))


if __name__ == "__main__":
    main()
