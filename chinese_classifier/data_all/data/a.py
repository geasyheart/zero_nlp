# -*- coding: utf8 -*-
#
import json

import pandas as pd


def to_json(file, to_file):
    to_f = open(to_file, 'w')
    df = pd.read_csv(file)
    for index, row in df.iterrows():
        label = int(row['label'])
        text = row['text']

        to_f.write(json.dumps({"label": label, "text": text}, ensure_ascii=False) + "\n")

    to_f.close()
to_json(
    '/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/data_all/data/test_data.csv',
    '/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/data_all/data/test_data.jsonl',
)
to_json(
    '/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/data_all/data/train_data.csv',
    '/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/data_all/data/train_data.jsonl',
)
to_json(
    '/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/data_all/data/valid_data.csv',
    '/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/data_all/data/valid_data.jsonl',
)