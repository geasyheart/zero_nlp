# -*- coding: utf8 -*-
#
import os

import sentencepiece as spm
import torch
from transformers import GPTNeoForCausalLM, GPT2LMHeadModel

path = '/home/yuzhang/windows_share/python-packages/pretrained/mengzi-gpt-neo-base'

tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(path, "mengzi_gpt.model"))
model = GPTNeoForCausalLM.from_pretrained(path)

# input_ids = torch.tensor(tokenizer.encode(['你和我']), dtype=torch.long, device='cpu')
input_ids = torch.tensor([[31, 7773, 31], [7773, 31, 31]])
outputs = model(**{"input_ids": input_ids}, labels=input_ids)
loss = outputs.loss
logits = outputs.logits
print()

# 本文件主要目的在于查看GPT是怎么计算loss的，由于GPTNeo计算方法和GPT一样，故以neo为例。
# 计算loss方式：
# 错开一个字，例如原句:
# input: 我 和 你
# shift_logits: 我 和
# shift_labels: 和 你
# 从而可以用交叉熵计算shift_logits和shift_labels的loss了
