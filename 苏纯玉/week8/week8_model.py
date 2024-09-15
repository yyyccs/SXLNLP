#第八周文本匹配， 使用cosine_triplet_loss损失函数进行训练
# 对于model.py老师所给源代码进行如下修改：
# def cosine_triplet_loss(self, a, p, n, margin=None):
#         ap = self.cosine_distance(a, p)
#         an = self.cosine_distance(a, n)
#         if margin is None:
#             diff = ap - an + 0.1
#         else:
#             diff = ap - an + margin.squeeze()
#         return torch.mean(diff[diff.gt(0)]) #greater than

#     #sentence : (batch_size, max_length)
#     def forward(self, sentence1, sentence2=None, target=None):
#         #同时传入两个句子
#         if sentence2 is not None:
#             vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
#             vector2 = self.sentence_encoder(sentence2)
#             #如果有标签，则计算loss
#             if target is not None:
#                 return self.loss(vector1, vector2, target.squeeze())
#             #如果无标签，计算余弦距离
#             else:
#                 return self.cosine_distance(vector1, vector2)
#         #单独传入一个句子时，认为正在使用向量化能力
#         else:
#             return self.sentence_encoder(sentence1)

# 修改为：
#     def cosine_triplet_loss(self, a, p, n, margin=None):
#         ap = self.cosine_distance(a, p)
#         an = self.cosine_distance(a, n)
#         if margin is None:
#             diff = ap - an + 0.1
#         else:
#             diff = ap - an + margin.squeeze()
#         return torch.mean(diff[diff.gt(0)]) #greater than

#     #sentence : (batch_size, max_length)
#     def forward(self, sentence1, sentence2=None, sentence3=None):
#         #同时传入两个句子
#         if sentence2 is not None:
#             vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
#             vector2 = self.sentence_encoder(sentence2)
#             vector3 = self.sentence_encoder(sentence3)
#             #如果有标签，则计算loss
#             if sentence3 is not None:
#                 return self.cosine_triplet_loss(vector1, vector2, vector3)
#             #如果无标签，计算余弦距离
#             else:
#                 return self.cosine_distance(vector1, vector2)
#         #单独传入一个句子时，认为正在使用向量化能力
#         else:
#             return self.sentence_encoder(sentence1)
