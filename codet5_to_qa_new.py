from __future__ import print_function

from tqdm import tqdm
import torch
import datasets
import transformers
import evaluate

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

##############
# PARAMETERS #
##############

T5_MODEL = "Salesforce/codet5-small"
BATCH_SIZE = 6
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
    train_data, val_data, test_data = data['train'], data['validation'], data['test']
    train_data = Dataset.from_pandas(data_formatter(train_data).dropna())
    train_data = train_data.select(range(0, 10000))
    val_data = Dataset.from_pandas(data_formatter(val_data).dropna())
    val_data = val_data.select(range(0, 2000))
    test_data = Dataset.from_pandas(data_formatter(test_data).dropna())
    # test_data = test_data.select(range(0, 50))

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

def evaluate_old(self, predictions, gold_answers):
        """_summary_

        Args:
            predictions (_type_): _description_
            gold_answers (_type_): _description_

        Returns:
            _type_: _description_
        """
        f1 = exact_match = 0

        for ground_truths, prediction in tqdm(zip(gold_answers, predictions)):
            # Remove pad token
            tokens_to_remove = {
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.mask_token_id
            }
            prediction = list(filter(lambda token: token not in tokens_to_remove, prediction))
            ground_truths = list(filter(lambda token: token not in tokens_to_remove, ground_truths))
            f1 += self.__f1_score(prediction, ground_truths)
            exact_match += self.__exact_match_score(prediction, ground_truths)
        return 100*f1/len(predictions), 100*exact_match/len(predictions)

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

def compute_metrics(p):
    predictions, gold_answers = p
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=2)
    predictions, gold_answers = predictions.tolist(), gold_answers.tolist()
    # print(f"\nlength of predictions: {len(predictions[0])}, length of gold_answers: {len(gold_answers[0])}")
    # print(f"\nexample: {predictions[0][0]}")
    # print(f"\ngold_example: {gold_answers[0][0]}")
    # print(f"prediction: {predictions[0]}, answer: {gold_answers[0]}")
    f1 = exact_match = 0
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    for ground_truths, prediction in tqdm(zip(gold_answers, predictions), desc="eval"):
        # Remove pad token
        tokens_to_remove = {
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.mask_token_id
        }
        prediction = list(filter(lambda token: token not in tokens_to_remove, prediction))
        ground_truths = list(filter(lambda token: token not in tokens_to_remove, ground_truths))
        f1 += f1_score(prediction, ground_truths)
        exact_match += exact_match_score(prediction, ground_truths)
    res = {}
    res['f1'], res['exact_match'] = f1, exact_match
    return res
    
if __name__ == "__main__": 

    _data = load_dataset("duorc", "SelfRC")
    model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_set, val_set, test_set = format_and_tokenize(_data)

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
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model('test_codet5_qa.model')
