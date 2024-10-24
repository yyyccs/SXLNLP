from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertTokenizer, BertModel
import re
from loguru import logger
logger.add('train.log', level='INFO')


def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位

    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True,
                         max_length=10)  # 将字转换成序号
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)

    return x, y


class torch_model(nn.Module):
    def __init__(self, bert_path, hidden_size=768, vocab_size=21128):
        super(torch_model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        if y is not None:
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)


def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位

    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True,
                         max_length=10)  # 将字转换成序号
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)

    return x, y

def load_text(path):
    text = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            text += line.strip()
    return text
def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
    return openings

def train():
    text_path = r"D:\download\BaiduNetdiskDownload\week\week10 文本生成问题\week10 文本生成问题\lstm语言模型生成文本\corpus.txt"
    text = load_text(text_path)
    bert_path = r"D:\download\BaiduNetdiskDownload\bert-base-chinese\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    epochs = 50
    batch_size = 256
    data_sample = 20000
    window_size = 10
    lr = 0.001
    model = torch_model(bert_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for batch in range(int(data_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, text)
            optimizer.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()
            watch_loss.append(loss.item())
        logger.info("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        logger.info(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        logger.info(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))

if __name__ == '__main__':
    train()
