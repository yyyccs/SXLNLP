import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import random

from loguru import logger

logger.add('sft.log', level='INFO')

config = {
    "bert_path": r"C:\Users\admin-HW\Desktop\入职\work\week\bert-base-chinese",
    "max_length": 128,
}


class DataGenerator:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(r"C:\Users\admin-HW\Desktop\入职\work\week\bert-base-chinese")
        self.data = []
        self.load_data()

    def load_data(self):
        with open('sample_data.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line_info = eval(line.replace('\n', '').replace(' ', ''))
                title = line_info['title']
                content = line_info['content']
                title_ids = self.tokenizer.encode(title, add_special_tokens=False)
                content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                input_ids = [self.tokenizer.cls_token_id] + title_ids + [self.tokenizer.sep_token_id] + content_ids + [
                    self.tokenizer.sep_token_id]
                label_ids = [-1 for _ in range(len(title_ids))] + [-1] + content_ids + [self.tokenizer.sep_token_id] + [
                    -1]
                x = input_ids[:config["max_length"]] + [0] * (config['max_length'] - len(input_ids))
                y = label_ids[:config["max_length"]] + [0] * (config["max_length"] - len(label_ids))
                mask = torch.cat([torch.ones(len(input_ids), (len(title_ids) + 2),dtype=torch.long), torch.cat(
                    [torch.zeros(len(title_ids) + 2, len(content_ids) + 1,dtype=torch.long),
                     torch.tril(torch.ones(len(content_ids) + 1, len(content_ids) + 1,dtype=torch.long))],
                    dim=0)], dim=-1)
                m = config["max_length"]
                self.data.append([torch.LongTensor(x), torch.LongTensor(y), torch.LongTensor(self.pad_mask(mask, (m, m)))])

    def pad_mask(self, tensor, target_shape):
        # 获取输入张量和目标形状的长宽
        height, width = tensor.shape
        target_height, target_width = target_shape
        # 创建一个全零张量,形状为目标形状
        result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
        h_end = min(height, target_height)
        w_end = min(width, target_width)
        # 将原始张量对应的部分填充到全零张量中
        result[:h_end, :w_end] = tensor[:h_end, :w_end]
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class torch_model(nn.Module):
    def __init__(self, bert_path, hidden_size=768, vocab_size=21128):
        super(torch_model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, y=None, mask=None):
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)


def main():
    tokenizer = BertTokenizer.from_pretrained(r"C:\Users\admin-HW\Desktop\入职\work\week\bert-base-chinese")
    epoch_num = 50
    batch_size = 32
    lr = 0.001
    dg = DataGenerator()
    train_loader = DataLoader(dg, batch_size=batch_size, shuffle=True)
    model = torch_model(config["bert_path"])
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epoch_num):
        watch_loss = []
        model.train()
        for x, y, mask in train_loader:
            model.zero_grad()
            loss = model(x, y=y, mask=mask)
            loss.backward()
            watch_loss.append(loss.item())
            optim.step()
        logger.info("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        logger.info(generate_sentence("北京明年拟推工作日半价观看电影", model, tokenizer))
        logger.info(generate_sentence("南京一合金厂锅炉发生爆炸", model, tokenizer))


def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
        return tokenizer.decode(openings)
    return tokenizer.decode(openings)


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


if __name__ == '__main__':
    main()
