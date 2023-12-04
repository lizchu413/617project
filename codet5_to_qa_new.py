from __future__ import print_function

from tqdm import tqdm
import torch
import datasets
import transformers
# import evaluate

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer, T5ForConditionalGeneration, T5Tokenizer, AdamW, set_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer

from typing import List, Tuple
from collections import Counter

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import pandas as pd

##############
# PARAMETERS #
##############

T5_MODEL = "Salesforce/codet5-small"
BATCH_SIZE = 16
EPOCHS = 3
SAVE_EVERY = 1
LR = 1e-4
WORKERS = 2
DEVICE = "cuda"
MAX_INPUT_LENGTH = 512
SEED = 617

###################
# DATA PROCESSING #
###################

def tokenize_data(data): 
    contexts, questions, answers = data['contexts'], data['questions'], data['answers']
    answers = [''.join(ele) for ele in answers]
    inputs = list(map(lambda tuple: f"question:{tuple[0]}  context:{tuple[1]}", zip(questions,contexts)))
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    encoded_inputs = tokenizer(
                            inputs,
                            padding="max_length",
                            max_length=MAX_INPUT_LENGTH,
                            truncation=True,
                            return_tensors="pt",
                        )
    encoded_targets = tokenizer(
                            answers,
                            padding="max_length",
                            max_length=MAX_INPUT_LENGTH,
                            truncation=True,
                            return_tensors="pt",
                        )
    input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
    encoded_targets = encoded_targets.input_ids

    # replace padding target token id's of the labels by -100, crossEntropy skip target label == -100
    encoded_targets[encoded_targets == tokenizer.pad_token_id] = -100

    res = {}
    res['input_ids'], res['attention_mask'], res['labels'] = input_ids, attention_mask, encoded_targets
    return res

def data_formatter(data): 
    res = []
    for row in data: 
        nr_answer = len(row["answers"])
        res.append([[row["plot"]]*nr_answer, [row["question"]]*nr_answer, [answer if len(answer) > 0 else "" for answer in row["answers"]]])
    return pd.DataFrame(res, columns=['contexts', 'questions', 'answers'])

def format_and_tokenize(data): 
    print(data['train'])
    train_data, val_data, test_data = data['train'], data['validation'], data['test']
    train_data = Dataset.from_pandas(data_formatter(train_data).dropna())
    val_data = Dataset.from_pandas(data_formatter(val_data).dropna())
    test_data = Dataset.from_pandas(data_formatter(test_data).dropna())

    train_tokenized = train_data.map(tokenize_data, batched=True)
    val_tokenized = val_data.map(tokenize_data, batched=True)
    test_tokenized = test_data.map(tokenize_data, batched=True)
    return train_tokenized, val_tokenized, test_tokenized

# class Trainer(transformers.Trainer): 

#     def training_step(self, model, inputs):
#         """
#         Perform a training step on a batch of inputs.

#         Subclass and override to inject custom behavior.

#         Args:
#             model (`nn.Module`):
#                 The model to train.
#             inputs (`Dict[str, Union[torch.Tensor, Any]]`):
#                 The inputs and targets of the model.

#                 The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
#                 argument `labels`. Check your model's documentation for all accepted arguments.

#         Return:
#             `torch.Tensor`: The tensor with training loss on this batch.
#         """
#         model.train()
#         loss = self.compute_loss(model, inputs)

#         if self.args.n_gpu > 1:
#             loss = loss.mean()  # mean() to average on multi-gpu parallel training

#         return loss.detach() / self.args.gradient_accumulation_steps
    
#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         How the loss is computed by Trainer. By default, all models return the loss in the first element.

#         Subclass and override for custom behavior.
#         """
#         print(inputs)
#         input_ids, attention_mask, encoded_targets = inputs
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=encoded_targets)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         epoch_train_loss += loss.item() * BATCH_SIZE
#         return (loss, outputs) if return_outputs else loss

if __name__ == "__main__": 

    _data = load_dataset("duorc", "SelfRC")
    model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_set, val_set, test_set = format_and_tokenize(_data)

    print(train_set[0])

    args = TrainingArguments(
        output_dir="codet5-to-qa", 
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
    )

    trainer = Trainer(
        model=model, 
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.evaluate()





