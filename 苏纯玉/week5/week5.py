#第五周作业
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

def load_title():
    with open('titles.txt', 'r', encoding='utf-8') as f:
        title_list =[i.strip() for i in f.readlines()]
        print(title_list[:3])
    return title_list

def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def title_to_vectors(title_list, model):
    vectors = []
    for title in title_list:
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in title:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(title))
    return np.array(vectors)

def main():
    model = load_word2vec_model("model.w2v")
    sentences = load_title()
    vectors = title_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()
