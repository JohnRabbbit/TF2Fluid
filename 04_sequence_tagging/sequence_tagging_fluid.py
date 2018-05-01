#!/usr/bin/env python
#coding=utf-8
import os
import math

import paddle.v2 as paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer

from config import ModelConfig
from data_reader_fluid import data_reader


class NER_net(object):
    def __init__(self, config):
        self.config = config
        self.__build_net(config)

    def __build_net(self, config):
        def __net_conf(word, target):
            x = fluid.layers.embedding(
                input=word,
                size=[config.src_vocab_size, config.embedding_dim],
                dtype="float32",
                is_sparse=True)
            fwd_lstm, _ = fluid.layers.dynamic_lstm(
                input=fluid.layers.fc(size=config.unit_num * 4, input=x),
                size=config.unit_num * 4,
                candidate_activation="tanh",
                gate_activation="sigmoid",
                cell_activation="sigmoid",
                bias_attr=fluid.ParamAttr(
                    initializer=NormalInitializer(loc=0.0, scale=1.0)),
                is_reverse=False)
            bwd_lstm, _ = fluid.layers.dynamic_lstm(
                input=fluid.layers.fc(size=config.unit_num * 4, input=x),
                size=config.unit_num * 4,
                candidate_activation="tanh",
                gate_activation="sigmoid",
                cell_activation="sigmoid",
                bias_attr=fluid.ParamAttr(
                    initializer=NormalInitializer(loc=0.0, scale=1.0)),
                is_reverse=True)
            outputs = fluid.layers.concat([fwd_lstm, bwd_lstm], axis=1)

            emission = fluid.layers.fc(size=config.tag_num, input=outputs)

            crf_cost = fluid.layers.linear_chain_crf(
                input=emission,
                label=target,
                param_attr=fluid.ParamAttr(name="crfw", ))
            avg_cost = fluid.layers.mean(x=crf_cost)
            return avg_cost, emission

        config = self.config

        self.source = fluid.layers.data(
            name="source", shape=[1], dtype="int64", lod_level=1)
        self.target = fluid.layers.data(
            name="target", shape=[1], dtype="int64", lod_level=1)

        if config.parallel:
            places = fluid.layers.get_places()
            pd = fluid.layers.ParallelDo(places)
            with pd.do():
                source_ = pd.read_input(self.source)
                target_ = pd.read_input(self.target)

                avg_cost, emission_base = __net_conf(source_, target_)

                pd.write_output(avg_cost)
                pd.write_output(emission_base)

            avg_cost_list, self.emission = pd()
            self.avg_cost = fluid.layers.mean(x=avg_cost_list)
            self.emission.stop_gradient = True
        else:
            self.avg_cost, self.emission = __net_conf(self.source, self.target)


def train(conf):
    net = NER_net(conf)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=1e-3)
    sgd_optimizer.minimize(net.avg_cost)

    crf_decode = fluid.layers.crf_decoding(
        input=net.emission, param_attr=fluid.ParamAttr(name="crfw"))

    chunk_evaluator = fluid.evaluator.ChunkEvaluator(
        input=crf_decode,
        label=net.target,
        chunk_scheme="IOB",
        num_chunk_types=int(math.ceil((conf.tag_num - 1) / 2.0)))

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        test_target = chunk_evaluator.metrics + chunk_evaluator.states
        inference_program = fluid.io.get_inference_program(test_target)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            data_reader(conf.train_src_file_name, conf.train_trg_file_name,
                        conf.src_vocab_file, conf.trg_vocab_file),
            buf_size=1024000),
        batch_size=conf.batch_size)

    place = fluid.CUDAPlace(0) if conf.use_gpu else fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[net.source, net.target], place=place)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    for pass_id in xrange(conf.epoch_num):
        chunk_evaluator.reset(exe)
        for batch_id, data in enumerate(train_reader()):
            cost, batch_precision, batch_recall, batch_f1_score = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[net.avg_cost] + chunk_evaluator.metrics)

            if batch_id and batch_id % 5 == 0:
                print(("Pass %d, Batch %d, Loss %.5f, "
                       "Precision %.5f, Recall %.5f, F1 %.5f") %
                      (pass_id, batch_id, cost[0], batch_precision[0],
                       batch_recall[0], batch_f1_score[0]))

        pass_precision, pass_recall, pass_f1_score = chunk_evaluator.eval(exe)
        print(("Pass %d, Loss %.5f, "
               "Precision %.5f, Recall %.5f, F1 %.5f") %
              (pass_id, cost[0], batch_precision[0], batch_recall[0],
               batch_f1_score[0]))
        fluid.io.save_inference_model(save_dirname,
                                      os.path.join(conf.fluid_model_path,
                                                   "params_pass_%d" % pass_id),
                                      ["source", "target"], [crf_decode], exe)


if __name__ == "__main__":
    conf = ModelConfig()
    train(conf)
