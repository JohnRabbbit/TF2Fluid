#!/usr/bin/env python
#coding=utf-8
import paddle.v2 as paddle
import paddle.fluid as fluid

from rnnlm_fluid import RNNLM
from load_data_fluid import train_data


class LMConfig(object):
    """Configuration of the RNN language model"""

    vocab_size = 10000
    batch_size = 64

    embedding_dim = 64
    hidden_dim = 128
    num_layers = 2
    rnn_model = "gru"

    learning_rate = 0.001

    parallel = True
    use_gpu = False

    num_passes = 5


def train():
    conf = LMConfig()
    word, lbl, _, avg_cost = RNNLM(conf).build_rnnlm()

    sgd_optimizer = fluid.optimizer.Adam(learning_rate=conf.learning_rate)
    sgd_optimizer.minimize(avg_cost)

    train_reader = paddle.batch(
        paddle.reader.shuffle(train_data, buf_size=51200),
        batch_size=conf.batch_size)

    place = fluid.CUDAPlace(0) if conf.use_gpu else fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[word, lbl], place=place)

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    print(fluid.default_main_program().to_string(True))
    for pass_id in xrange(conf.num_passes):
        for batch_id, data in enumerate(train_reader()):
            cost = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost])
            print("Pass %d, Batch %d, Loss %.4f" % (pass_id, batch_id,
                                                    cost[0]))


if __name__ == "__main__":
    train()
