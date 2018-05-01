# -*- coding: utf-8 -*-
import os


class ModelConfig(object):
    job = "train"

    train_src_file_name = "data/train_src.txt"
    src_vocab_file = "data/train_src.vocab"
    src_vocab_size = len(open(src_vocab_file, "r").readlines()) - 2

    train_trg_file_name = "data/train_trg.txt"
    trg_vocab_file = "data/train_trg.vocab"
    tag_num = len(open(trg_vocab_file).readlines()) - 2

    dev_src_file_name = "data/dev_src.txt"
    dev_trg_file_name = "data/dev_trg.txt"

    batch_size = 64 if job == "train" else 1

    embedding_dim = unit_num = 128
    max_sequence = 100
    drop_rate = 0.5 if job == "train" else 1.

    epoch_num = 10

    tf_model_path = "tf_models"
    if not os.path.exists(tf_model_path): os.mkdir(tf_model_path)
    """
    fluid configurations
    """
    parallel = True
    use_gpu = False
    fluid_model_path = "fluid_models"
    if not os.path.exists(fluid_model_path): os.mkdir(fluid_model_path)
