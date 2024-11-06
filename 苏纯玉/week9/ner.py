import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from TorchCRF import CRF
from loguru import logger
from transformers import BertTokenizer, BertModel
from loguru import logger
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import re

logger.add("log.txt", level="INFO")


class torch_model(nn.Module):
    def __init__(self, bert_path, hidden_size, class_num):
        super(torch_model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        # self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None):
        x, _ = self.bert(x)
        predict = self.classify(x)
        if target is not None:
            # return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
            mask = target.gt(-1)
            return - self.crf_layer(predict, target, mask, reduction="mean")
        else:
            return predict


def pad_tun(input_ids, label_ids, max_length):
    if len(input_ids) < max_length:
        input_ids.extend([0] * (max_length - len(input_ids)))
        label_ids.extend([0] * (max_length - len(label_ids)))
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        label_ids = label_ids[:max_length]
    return [torch.LongTensor(input_ids), torch.LongTensor(label_ids)]


def load_data(path, tokensizer, schema, max_length):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for segment in f.read().split('\n\n'):
            sentence_inputids = []
            sentence_lable = []
            for line in segment.split("\n"):
                if line.strip() == "":
                    continue
                char, label = line.split()
                sentence_inputids.append(tokensizer.encode(text=char, add_special_tokens=False)[0])
                sentence_lable.append(schema[label])
            data.append(pad_tun(sentence_inputids, sentence_lable, max_length))
    return data

def main():
    bert_path = r"D:\download\BaiduNetdiskDownload\bert-base-chinese\bert-base-chinese"
    test_path = r"D:\download\BaiduNetdiskDownload\week\week9 序列标注问题\week9 序列标注问题\ner\ner_data\test"
    train_path = r"D:\download\BaiduNetdiskDownload\week\week9 序列标注问题\week9 序列标注问题\ner\ner_data\train"
    max_length = 256
    schema = {
        "B-LOCATION": 0,
        "B-ORGANIZATION": 1,
        "B-PERSON": 2,
        "B-TIME": 3,
        "I-LOCATION": 4,
        "I-ORGANIZATION": 5,
        "I-PERSON": 6,
        "I-TIME": 7,
        "O": 8
    }

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = torch_model(bert_path, hidden_size=768, class_num=len(schema))
    data = load_data(train_path, tokenizer, schema, max_length)
    epoch_num = 10
    batch_size = 128
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epoch_num):
        epoch += 1
        model.train()
        train_loss = []
        for item in DataLoader(data, batch_size=batch_size, shuffle=True):
            train_x, train_y = item
            optimizer.zero_grad()  # 梯度归零
            loss = model(train_x, train_y)
            loss.backward()  # 计算梯度
            optimizer.step()
            train_loss.append(loss.item())
        logger.info("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(train_loss)))
        model.eval()
        t = 0
        f = 0
        with torch.no_grad():
            test_data = load_data(test_path, tokenizer, schema, max_length)
            for item in DataLoader(test_data, batch_size=batch_size, shuffle=False):
                test_x, test_y = item
                test_y = test_y.numpy()
                output = model(test_x)
                output = torch.argmax(output, dim=-1).numpy()
                mask_t = test_y == output
                mask_f = test_y != output
                t = t+np.sum(mask_t)
                f = f+np.sum(mask_f)
        logger.info(f"=========\n第%d轮平准确率{(t/(t+f))*100}%")



if __name__ == '__main__':
    main()
