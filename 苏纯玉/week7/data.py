import json
import random

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('文本分类练习.csv')
dic = df.set_index('review')['label'].to_dict()

# 将数据集分离为正负样本
positive_samples = [(k, v) for k, v in dic.items() if v == 1]
negative_samples = [(k, v) for k, v in dic.items() if v == 0]

# 对正负样本分别进行分层抽样
train_positive, test_positive = train_test_split(positive_samples, test_size=0.3, random_state=42)
train_negative, test_negative = train_test_split(negative_samples, test_size=0.3, random_state=42)

# 合并训练集和测试集
train_set = train_positive + train_negative
test_set = test_positive + test_negative

# 转换回字典格式
train_set_dict = [{'review': k, 'label': v} for k, v in train_set]
test_set_dict = [{'review': k, 'label': v} for k, v in test_set]
# 打乱数据顺序
random.shuffle(train_set_dict)
random.shuffle(test_set_dict)

# 将数据写入JSON文件
with open('..data/train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_set_dict, f, ensure_ascii=False, indent=4)
with open('..data/test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_set_dict, f, ensure_ascii=False, indent=4)