from __future__ import print_function

from tqdm import tqdm
import torch
import datasets
import transformers

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
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
torch.cuda.empty_cache()

##############
# PARAMETERS #
##############

T5_MODEL = "Salesforce/codet5p-220m"
BATCH_SIZE = 8
EPOCHS = 3
SAVE_EVERY = 1
LR = 1e-4
WORKERS = 2
DEVICE = "cuda"
MAX_INPUT_LENGTH = 120
SEED = 617

###################
# DATA PROCESSING #
###################

tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)

def tokenize_data(data): 
    contexts, questions, answers = data['contexts'], data['questions'], data['answers']
    answers = [''.join(ele) for ele in answers]
    inputs = list(map(lambda tuple: f"question:{tuple[0]}  context:{tuple[1]}", zip(questions,contexts)))
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
    counter = 0
    for row in tqdm(data):
        res.append([(' '.join(row['func_code_tokens'])), row['question'], (' '.join(row['answer']))])
        # if counter == 0: 
        #     print(row)
        #     print("==== res ====")
        #     print(res)
        counter += 1
    return pd.DataFrame(res, columns=['contexts', 'questions', 'answers'])

def format_and_tokenize(data): 
    data = data.filter(lambda x : x["question"] == "What does this function do?")
    train_data, val_data, test_data = data['train'], data['validation'], data['test']
    train_data = train_data.take(10000)
    train_data = Dataset.from_pandas(data_formatter(train_data).dropna())
    val_data = val_data.take(2000)
    val_data = Dataset.from_pandas(data_formatter(val_data).dropna())
    test_data = test_data.take(2000)
    test_data = Dataset.from_pandas(data_formatter(test_data).dropna())

    train_tokenized = train_data.map(tokenize_data, batched=True)
    val_tokenized = val_data.map(tokenize_data, batched=True)
    test_tokenized = test_data.map(tokenize_data, batched=True)
    return train_tokenized, val_tokenized, test_tokenized

def exact_match_score(prediction, ground_truth):
    if len(ground_truth) == len(prediction):
        if all(token1 == token2 for token1, token2 in zip(ground_truth,prediction)):
            return 1
    return 0

def f1_score(prediction_tokens, ground_truth_tokens):
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_metrics(p, tokens_to_remove):
    predictions, gold_answers = p
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=2)
    predictions, gold_answers = predictions.tolist(), gold_answers.tolist()
    f1 = exact_match = 0
    for ground_truths, prediction in tqdm(zip(gold_answers, predictions), desc="eval"):
        prediction = list(filter(lambda token: token not in tokens_to_remove, prediction))
        ground_truths = list(filter(lambda token: token not in tokens_to_remove, ground_truths))
        f1 += f1_score(prediction, ground_truths)
        exact_match += exact_match_score(prediction, ground_truths)
    res = {}
    res['f1'], res['exact_match'] = f1, exact_match
    return res
    
if __name__ == "__main__": 

    _data = load_dataset("aalexchengg/codesearchnet_qa", streaming = True)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL, trust_remote_code=True)

    tokens_to_remove = {
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.mask_token_id
        }

    train_set, val_set, test_set = format_and_tokenize(_data)

    args = TrainingArguments(
        output_dir="codet5-to-qa", 
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        load_best_model_at_end=True,
        eval_accumulation_steps=1,
        fp16 = True
    )

    trainer = Trainer(
        model=model, 
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model('test_codet5_qa.model')
