# -*- coding: utf8 -*-
#
import json
from typing import Optional

import torch
from sklearn.metrics import f1_score, classification_report
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, BertModel
from transformers import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

MODEL_NAME_OR_PATH = "/home/yuzhang/PycharmProjects/zero_nlp/chinese_classifier/model_result/checkpoint-800"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)


class MyModel(BertPreTrainedModel):
    def __init__(self, config):
        super(MyModel, self).__init__(config)
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.num_labels = 2
        self.dp = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.post_init()

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


model = MyModel.from_pretrained(MODEL_NAME_OR_PATH)

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
