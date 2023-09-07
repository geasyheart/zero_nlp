# -*- coding: utf8 -*-
#
import shutil

import evaluate
import numpy as np
import transformers
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

metric = evaluate.load("accuracy")
# "bert-base-chinese"  # "distilbert-base-uncased"
MODEL_NAME_OR_PATH = "/home/yuzhang/windows_share/python-packages/pretrained/chinese-roberta-wwm-ext"
NUM_LABELS = 2
MAX_LENGTH = 64

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_OR_PATH, num_labels=NUM_LABELS)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

text_dataset = load_dataset(
    'json', data_files={
        'train': ['data_all/data/train_data.jsonl'],
        'test': ['data_all/data/test_data.jsonl']},
)


def preprocess_function(examples):
    res = tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)
    return res


# 在实际工程中，会先使用`Tokenizer`把所有的文本转换成`input_ids`,`token_type_ids`,`attention_mask`，然后在训练的时候，这步就不再做了，目的是减少训练过程中cpu处理数据的时间，不给显卡休息时间。
tokenized_text = text_dataset.map(preprocess_function, batched=True,)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, max_length=MAX_LENGTH)

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
    train_dataset=tokenized_text["train"],
    eval_dataset=tokenized_text["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
