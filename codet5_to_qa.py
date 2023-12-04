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

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

##############
# PARAMETERS #
##############

T5_MODEL = "Salesforce/codet5-small"
BATCH_SIZE = 1
EPOCHS = 3
SAVE_EVERY = 1
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

###################    
# MULTI-GPU SETUP #
###################

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        tokenizer, 
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, contexts, questions, answers, epoch_train_loss):
        # self.optimizer.zero_grad()
        # output = self.model(source)
        # loss = F.cross_entropy(output, targets)
        # loss.backward()
        # self.optimizer.step()
        self.optimizer.zero_grad()
        inputs = list(map(lambda tuple: f"question:{tuple[0]}  context:{tuple[1]}", zip(questions, contexts)))
        encoded_inputs = self.tokenizer(
                                inputs,
                                padding="longest",
                                max_length=MAX_INPUT_LENGTH,
                                truncation=True,
                                return_tensors="pt",
                            )
        encoded_targets = self.tokenizer(
                                answers,
                                padding="longest",
                                max_length=MAX_INPUT_LENGTH,
                                truncation=True,
                                return_tensors="pt",
                            )

        input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
        encoded_targets = encoded_targets.input_ids

        # replace padding target token id's of the labels by -100, crossEntropy skip target label == -100
        encoded_targets[encoded_targets == self.tokenizer.pad_token_id] = -100

        input_ids = input_ids.to(self.gpu_id)
        encoded_targets = encoded_targets.to(self.gpu_id)
        attention_mask = attention_mask.to(self.gpu_id)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=encoded_targets)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        epoch_train_loss += loss.item() * BATCH_SIZE
        return epoch_train_loss

    def _run_epoch(self, epoch):
        # b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.train_data)}")
        # self.train_data.sampler.set_epoch(epoch)
        # for source, targets in self.train_data:
        #     source = source.to(self.gpu_id)
        #     targets = targets.to(self.gpu_id)
        #     self._run_batch(source, targets)

        self.train_data.sampler.set_epoch(epoch)

        epoch_train_loss = 0.
        for context, questions, answers in tqdm(self.train_data): 
            context = context.to(self.gpu_id)
            questions = questions.to(self.gpu_id)
            answers = answers.to(self.gpu_id)
            epoch_train_loss = self._run_batch(context, questions, answers, epoch_train_loss)

        print(f"epoch={epoch + 1}")
        print(f"\t Train loss = {epoch_train_loss/len(self.train_data):.4f}")

        # EVALUATION STEP 
        self.model.eval()
        with torch.no_grad():
            model_predictions_encoded = []
            target_encoded = []
            for contexts, questions, answers in tqdm(self.val_data):
                inputs = list(map(lambda tuple: f"question: {tuple[0]}  context:{tuple[1]}", zip(
                    questions, contexts)))
                encoded_inputs = self.tokenizer(
                    inputs,
                    padding="longest",
                    max_length=MAX_INPUT_LENGTH,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded_targets = self.tokenizer(
                    answers,
                    padding="longest",
                    max_length=MAX_INPUT_LENGTH,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded_inputs, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
                encoded_targets = encoded_targets.input_ids

                encoded_inputs = encoded_inputs.to(self.gpu_id)
                encoded_targets = encoded_targets.to(self.gpu_id)
                attention_mask = attention_mask.to(self.gpu_id)
                model_predictions = self.model.generate(
                    input_ids=encoded_inputs, attention_mask=attention_mask)

                model_predictions_encoded += model_predictions.tolist()
                target_encoded += encoded_targets.tolist()

        # DONE EVALUATING
        f1, exact_match = self.val_set.evaluate(model_predictions_encoded, target_encoded)

        print(f"\t Validation F1 = {f1:.2f}, EM = {exact_match:.2f}")
        if f1 > f1_old and self.gpu_id == 0 and epoch % self.save_every == 0:
            self.model.save_pretrained(f'results/{self.model.name_or_path}/model/best-f1')
            self.tokenizer.save_pretrained(f'results/{self.model.name_or_path}/tokenizer/best-f1')
            f1_old = f1
        if epoch+1 % 10 == 0 and self.gpu_id == 0 and epoch % self.save_every == 0:
            self.model.save_pretrained(f'results/{self.model.name_or_path}/model/checkpoint-{epoch+1}')
            self.tokenizer.save_pretrained(f'results/{self.tokenizer.name_or_path}/tokenizer/checkpoint-{epoch+1}')
        self.model.train()

    # def _save_checkpoint(self, epoch):
    #     ckp = self.model.module.state_dict()
    #     PATH = "checkpoint.pt"
    #     torch.save(ckp, PATH)
    #     print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            # if self.gpu_id == 0 and epoch % self.save_every == 0:
            #     self._save_checkpoint(epoch)

        if self.gpu_id == 0: 
            self.model.save_pretrained(
                f'results/{self.model.name_or_path}/model/checkpoint-{epoch+1}')
            self.tokenizer.save_pretrained(
                f'results/{self.model.name_or_path}/tokenizer/checkpoint-{epoch+1}')


def load_train_objs():
    _data = load_dataset("duorc", "SelfRC")

    model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_set = Dataset(_data["train"], tokenizer, parser=DatasetMap.duorc)
    val_set = Dataset(_data["validation"], tokenizer, parser=DatasetMap.duorc)

    return train_set, val_set, model, tokenizer, optimizer

def wrapper(dataset):
    def inside(data):
        return dataset.pack_minibatch(data)
    return inside

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=WORKERS,
        collate_fn=wrapper(dataset),
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    train_set, val_set, model, tokenizer, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_set, batch_size)
    val_data = prepare_dataloader(val_set, batch_size)
    trainer = Trainer(model, train_data, val_data, optimizer, 
                      tokenizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__": 
    # Set seed
    print("welcome to code-t5 to qa")
    set_seed(SEED)

    _data = load_dataset("duorc", "SelfRC")

    # model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    # # creating the optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # train_set = Dataset(_data["train"], tokenizer, parser=DatasetMap.duorc)
    # validation_set = Dataset(_data["validation"], tokenizer, parser=DatasetMap.duorc)

    # train(model=model,
    #     tokenizer=tokenizer,
    #     optimizer=optimizer,
    #     train_set=train_set,
    #     validation_set=validation_set,
    #     num_train_epochs=EPOCHS, device=DEVICE, batch_size=BATCH_SIZE)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, SAVE_EVERY, EPOCHS, BATCH_SIZE), nprocs=world_size)
