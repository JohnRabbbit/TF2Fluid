#!/usr/bin/env python
#coding=utf-8
import sys

import paddle.v2 as paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer

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

    parallel = False
    use_gpu = False

    num_passes = 5


class RNNLM(object):
    def __init__(self, config):
        self.parallel = config.parallel
        self.vocab_size = config.vocab_size

        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.rnn_model = config.rnn_model

        self.learning_rate = config.learning_rate

    def __input_embedding(self, onehot_word):
        return fluid.layers.embedding(
            input=onehot_word,
            size=[self.vocab_size, self.embedding_dim],
            dtype="float32",
            is_sparse=True)

    def __rnn(self, input):
        for i in range(self.num_layers):
            hidden = fluid.layers.fc(
                size=self.hidden_dim * 4,
                bias_attr=fluid.ParamAttr(
                    initializer=NormalInitializer(loc=0.0, scale=1.0)),
                input=hidden if i else input)
            lstm = fluid.layers.dynamic_lstm(
                input=hidden,
                size=self.hidden_dim * 4,
                candidate_activation="tanh",
                gate_activation="sigmoid",
                cell_activation="sigmoid",
                bias_attr=fluid.ParamAttr(
                    initializer=NormalInitializer(loc=0.0, scale=1.0)),
                is_reverse=False)
        return lstm

    def __cost(self, lstm_output, lbl):
        prediction = fluid.layers.fc(
            input=lstm_output, size=self.vocab_size, act="softmax")
        return prediction, fluid.layers.cross_entropy(
            input=prediction, label=lbl)

    def __network(self, word, lbl):
        word_embedding = self.__input_embedding(word)
        lstm_output = self.__rnn(word_embedding)
        prediction, cost = self.__cost(lstm_output, lbl)
        return prediction, fluid.layers.mean(x=cost)

    def build_rnnlm(self):
        word = fluid.layers.data(
            name="current_word", shape=[1], dtype="int64", lod_level=1)
        lbl = fluid.layers.data(
            name="next_word", shape=[1], dtype="int64", lod_level=1)

        if self.parallel:
            places = fluid.layers.get_places()
            pd = fluid.layers.ParallelDo(places)

            with pd.do():
                word_ = pd.read_input(word)
                lbl_ = pd.read_input(lbl)
                prediction, avg_cost = self.__network(word, lbl)
                pd.write_output(avg_cost)
                pd.write_output(prediction)
            prediction, avg_cost = pd()
            avg_cost = fluid.layers.mean(x=avg_cost)
        else:
            prediction, avg_cost = self.__network(word, lbl)
        return word, lbl, prediction, avg_cost


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
