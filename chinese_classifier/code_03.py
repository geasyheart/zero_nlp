# -*- coding: utf8 -*-
#
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
#
# model = AutoModelForSequenceClassification.from_pretrained(
#     '/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/model_result/checkpoint-800/',
#     num_labels=2)
#
# tokenizer = AutoTokenizer.from_pretrained('/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/model_result/checkpoint-800/')
# print(model.forward(**tokenizer('这个酒店太差了', return_tensors='pt')))
# print(model.forward(**tokenizer('这个酒店太好了', return_tensors='pt')))
# print(model.forward(**tokenizer('这个酒店太好了，但是服务很差', return_tensors='pt')))
import json

import torch
from sklearn.metrics import f1_score, classification_report
from transformers import pipeline

pipe = pipeline(
    task="text-classification",
    model="model_result/checkpoint-800",
    device=torch.device("cuda:0")
)
# test_str = ['这个酒店也太差了', '非常实惠']
# res = pipe(test_str)
# print(res)

preds, trues = [], []
with open('/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/data_all/data/test_data.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        res = pipe(data['text'][:510])
        preds.append(int(res[0]['label'].split('_')[-1]))
        trues.append(data['label'])

res = f1_score(trues, preds, average='macro')
print(classification_report(trues, preds))
