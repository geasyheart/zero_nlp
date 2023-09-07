# -*- coding: utf8 -*-
#
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from torch.utils.data import Dataset
import shutil

import evaluate
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

MODEL_NAME_OR_PATH = "/home/yuzhang/windows_share/python-packages/pretrained/chinese-roberta-wwm-ext"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        # output = tokenizer(data['text'], max_length=64, truncation=True)
        # output.update({"label": data['label']})
        # return output
        return data


train_dataset = MyDataSet(
    data=list(read_json('/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/data_all/data/train_data.jsonl')))
dev_dataset = MyDataSet(
    data=list(read_json('/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/data_all/data/test_data.jsonl')))


@dataclass
class MyDataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]):
        texts, labels = [], []
        for feature in features:
            texts.append(feature['text'])
            labels.append(feature['label'])
        bert_input = tokenizer(texts, max_length=64, truncation=True, padding=True, return_tensors='pt')
        labels = torch.LongTensor(labels)
        bert_input.update({"labels": labels})
        return bert_input


data_collator = MyDataCollator(tokenizer=tokenizer, max_length=64)
metric = evaluate.load("accuracy")

NUM_LABELS = 2
MAX_LENGTH = 64
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_OR_PATH, num_labels=NUM_LABELS)

LOGGING_DIR = "logging_dir"
MODEL_DIR = "model_result"
shutil.rmtree(LOGGING_DIR, ignore_errors=True)

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    overwrite_output_dir=True,
    logging_dir=LOGGING_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=32,  # 每一次batch，训练的数据的数量，如果显存高，可以32起步，如果一般，那可能就是个位数，比如2，4，8，16等。
    per_device_eval_batch_size=32,  # 在评估的时候，batch的大小，看显存大小了。
    do_eval=True,
    evaluation_strategy="steps",
    # eval_accumulation_steps=50,
    # evaluation_strategy='epoch',
    eval_steps=50,
    logging_steps=50,
    save_steps=100,
    num_train_epochs=4,  # 训练多少轮
    weight_decay=0.01,
    save_total_limit=3,  # 模型每`eval_steps`步，就会保存一下模型，只会保存最新的3个模型，
    jit_mode_eval=True,
    fp16=True,
    fp16_opt_level='O3',
    load_best_model_at_end=True,  # 最后，加载效果最好的模型
    remove_unused_columns=False
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    res = metric.compute(predictions=predictions, references=labels)

    res = f1_score(labels, predictions, average='macro')
    print(classification_report(labels, predictions))
    return {"macro_f1": res}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
