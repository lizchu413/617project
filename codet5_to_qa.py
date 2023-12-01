from __future__ import print_function

from tqdm import tqdm
import torch
import datasets
import transformers

from datasets import load_dataset
from transformers import PreTrainedTokenizer, T5ForConditionalGeneration, T5Tokenizer, AdamW, set_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from typing import List, Tuple
from collections import Counter

##############
# PARAMETERS #
##############

T5_MODEL = "Salesforce/codet5-small"
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4
WORKERS = 2
DEVICE = "cuda"
MAX_INPUT_LENGTH = 512
SEED = 617

#############################
# DATASET CLASS OVERWRITING #
#############################

class DatasetMap():
    @staticmethod
    def duorc(example):
        nr_answer = len(example["answers"])
        return [example["plot"]]*nr_answer, [example["question"]]*nr_answer, [answer if len(answer) > 0 else "" for answer in example["answers"]]

    @staticmethod
    def squad(example):
        nr_answer = len(example["answers"]["text"])
        return [example["context"]]*nr_answer, [example["question"]]*nr_answer, [answer if len(answer) > 0 else "" for answer in example["answers"]["text"]]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset: datasets.arrow_dataset.Dataset, tokenizer, parser=None):
        """Constructor for Dataset class
        Args:
            hf_dataset (datasets.arrow_dataset.Dataset): HuggingFace Dataset
            tokenizer: HuggingFace Tokenizer

        Raises:
            Exception: if two between questions, answers and contexts have different length it will raise an exception
        """
        self.tokenizer = tokenizer
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.contexts: List[str] = []

        for row in tqdm(hf_dataset):
            _contexts, _questions, _answers = parser(row)

            self.contexts += _contexts
            self.questions += _questions
            self.answers += _answers

        if len(self.questions) != len(self.answers) or len(self.questions) != len(self.contexts):
            raise Exception(
                "something wrong while building the dataset: questions, contexts and answers result in different dimensions")

        self.item_count: int = len(self.questions)

    def __len__(self):
        """Magic method over-ride for class lenght evaluation

        Returns:
            int: lenght of the object
        """
        return self.item_count

    def __getitem__(self, index: int):
        """Magic method over-ride for class getitem method

        Args:
            index (int): index for identify question-context and answer example

        Returns:
            Tuple(str,str,str): (Context, Question, Answer)
        """
        return self.contexts[index], self.questions[index], self.answers[index]

    def pack_minibatch(self, data: List[Tuple[str, str]]):
        """Pack mini-batch function

        Args:
            data (Tuple[List[str],List[str],List[str]]): (Contexts, Questions, Answers)

        Returns:
            Tuple[List[str],List[str],List[str]]: (Contexts, Questions, Answers)
        """
        return zip(*data)

    def __exact_match_score(self, prediction, ground_truth):
        """_summary_

        Args:
            prediction (_type_): _description_
            ground_truth (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(ground_truth) == len(prediction):
            if all(token1 == token2 for token1, token2 in zip(ground_truth,prediction)):
                return 1
        return 0

    def __f1_score(self, prediction_tokens, ground_truth_tokens):
        """_summary_

        Args:
            prediction (_type_): _description_
            ground_truth (_type_): _description_

        Returns:
            _type_: _description_
        """
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def evaluate(self, predictions, gold_answers):
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
    
#################
# TRAINING LOOP #
#################

def train(model: T5ForConditionalGeneration, tokenizer: PreTrainedTokenizer,
          optimizer: AdamW, train_set: Dataset, validation_set: Dataset,
          num_train_epochs: int, device: str, batch_size: int,
          max_input_length: int = 512):
    """_summary_

    Args:
        model (T5ForConditionalGeneration): _description_
        tokenizer (PreTrainedTokenizer): _description_
        optimizer (AdamW): _description_
        train_set (Dataset): _description_
        validation_set (Dataset): _description_
        num_train_epochs (int): _description_
        device (str): _description_
        batch_size (int): _description_
    """
    my_trainset_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE,
                                        num_workers=WORKERS, collate_fn=lambda data: train_set.pack_minibatch(data))
    my_validation_dataloader = DataLoader(validation_set, batch_size=BATCH_SIZE,
                                          num_workers=WORKERS, collate_fn=lambda data: validation_set.pack_minibatch(data))

    # set training mode on the model
    model.train()

    # model to device
    model.to(device)

    f1_old: int = 0
    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.
        for contexts,questions,answers in tqdm(my_trainset_dataloader):
            optimizer.zero_grad()

            inputs = list(map(lambda tuple: f"question:{tuple[0]}  context:{tuple[1]}", zip(questions,contexts)))
            encoded_inputs = tokenizer(
                                    inputs,
                                    padding="longest",
                                    max_length=max_input_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )
            encoded_targets = tokenizer(
                                    answers,
                                    padding="longest",
                                    max_length=max_input_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )

            input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
            encoded_targets = encoded_targets.input_ids

            # replace padding target token id's of the labels by -100, crossEntropy skip target label == -100
            encoded_targets[encoded_targets == tokenizer.pad_token_id] = -100

            input_ids = input_ids.to(device)
            encoded_targets = encoded_targets.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=encoded_targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_size
        print(f"epoch={epoch + 1}/{num_train_epochs}")
        print(f"\t Train loss = {epoch_train_loss/len(train_set):.4f}")

        model.eval()
        with torch.no_grad():
            model_predictions_encoded = []
            target_encoded = []
            for contexts, questions, answers in tqdm(my_validation_dataloader):
                inputs = list(map(lambda tuple: f"question: {tuple[0]}  context:{tuple[1]}", zip(
                    questions, contexts)))
                encoded_inputs = tokenizer(
                    inputs,
                    padding="longest",
                    max_length=max_input_length,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded_targets = tokenizer(
                    answers,
                    padding="longest",
                    max_length=max_input_length,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded_inputs, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
                encoded_targets = encoded_targets.input_ids

                encoded_inputs = encoded_inputs.to(device)
                encoded_targets = encoded_targets.to(device)
                attention_mask = attention_mask.to(device)
                model_predictions = model.generate(
                    input_ids=encoded_inputs, attention_mask=attention_mask)

                model_predictions_encoded += model_predictions.tolist()
                target_encoded += encoded_targets.tolist()
        f1, exact_match = validation_set.evaluate(model_predictions_encoded, target_encoded)

        print(f"\t Validation F1 = {f1:.2f}, EM = {exact_match:.2f}")
        if f1 > f1_old :
            model.save_pretrained(f'results/{model.name_or_path}/model/best-f1')
            tokenizer.save_pretrained(f'results/{model.name_or_path}/tokenizer/best-f1')
            f1_old = f1
        if epoch+1 % 10 == 0:
            model.save_pretrained(f'results/{model.name_or_path}/model/checkpoint-{epoch+1}')
            tokenizer.save_pretrained(f'results/{model.name_or_path}/tokenizer/checkpoint-{epoch+1}')
        model.train()

    model.save_pretrained(
        f'results/{model.name_or_path}/model/checkpoint-{epoch+1}')
    tokenizer.save_pretrained(
        f'results/{model.name_or_path}/tokenizer/checkpoint-{epoch+1}')
    


if __name__ == "__main__": 
    # Set seed
    print("welcome to code-t5 to qa")
    set_seed(SEED)

    _data = load_dataset("duorc", "SelfRC")

    model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    # creating the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_set = Dataset(_data["train"], tokenizer, parser=DatasetMap.duorc)
    validation_set = Dataset(_data["validation"], tokenizer, parser=DatasetMap.duorc)

    train(model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_set=train_set,
        validation_set=validation_set,
        num_train_epochs=EPOCHS, device=DEVICE, batch_size=BATCH_SIZE)
