# 苏纯玉week3作业
import torch
import torch.nn as nn
import numpy as np
import random
import json


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.embedding = nn.Embedding(28, 25)
        self.pool = nn.AvgPool1d(8)
        self.layer = nn.RNN(25, 1, bias=False, batch_first=True)
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss
    def forward(self, x, y = None):
        x = self.embedding(x)
        x = x.transpose_(1, 2)
        x = self.pool(x)
        x = x.squeeze()
        output, x = self.layer(x)
        y_pred = self.activation(output)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab

vocab = build_vocab()
def build_sample(sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if ('u' in x )or ('f'  in x )or ('o' in x):
        y = 1
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(sample_length,sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

def eval(model):
    model.eval()
    x, y = build_dataset(200, 8)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), 200 - sum(y)))# 建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        # x, y = build_dataset(200,8)  # 建立200个用于测试的样本
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print(correct, wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 25  # 每个字的维度
    sentence_length = 8  # 样本文本长度
    learning_rate = 0.008  # 学习率
    model = net()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample/batch_size)):
            x, y = build_dataset(batch_size, sentence_length) #x 20*8
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = eval(model)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
main()
