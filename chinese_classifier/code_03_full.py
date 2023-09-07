# -*- coding: utf8 -*-
#
from torch import nn
from transformers import BertModel
import json
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import evaluate
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertModel
from transformers import PreTrainedTokenizerBase, AutoModel
from transformers import TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput

MODEL_NAME_OR_PATH = "/home/yuzhang/windows_share/python-packages/pretrained/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.num_labels = 2
        self.bert = BertModel.from_pretrained(MODEL_NAME_OR_PATH)
        self.dp = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]

        pooled_output = self.dp(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


model = MyModel()
model.load_state_dict(torch.load('/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/model_result/checkpoint-800/pytorch_model.bin'))
model.eval()

with torch.no_grad():
    preds, trues = [], []
    with open('/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/data_all/data/test_data.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)

            output = model(**tokenizer(data['text'], max_length=64, truncation=True, padding=True, return_tensors='pt'))
            res = torch.softmax(output.logits[0], dim=-1).argmax().tolist()
            preds.append(res)
            trues.append(data['label'])

    res = f1_score(trues, preds, average='macro')
    print(classification_report(trues, preds))
