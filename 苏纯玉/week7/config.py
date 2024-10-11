# -*- coding: utf-8 -*-

"""
配置参数信息
"""
import json

Config = {
    "model_path": "output",
    "train_data_path": r"..\文本分类练习数据集\train_data.json",
    "valid_data_path": r"..\文本分类练习数据集\test_data.json",
    "vocab_path":"chars.txt",
    # "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"C:\Users\admin-HW\Desktop\Chrome109_AllNew_2024.8.31\w6\bert-base-chinese\bert-base-chinese",
    "seed": 987
}