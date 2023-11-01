import os, re, csv, time
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass
import torch
import random
import numpy as np
import argparse
from torch.utils.data import SequentialSampler, BatchSampler
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForLanguageModeling,
    RobertaForMaskedLM,
    GPT2LMHeadModel,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 10617
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# DCNL stands for newline
# DCSP stands for a space that is either in the leading identation of a line 
# (one token per nesting level) or inside a string constant
    
# load data or something like that
def data(): 
    with open("code-docstring-corpus-V2/repo_split/repo_split.parallel_desc.train", 
            "r", encoding="utf-8", errors='ignore') as file: 
        train_desc = [line[1:-2] for line in file]

    with open("code-docstring-corpus-V2/repo_split/repo_split.parallel_desc.test", 
            "r", encoding="utf-8", errors='ignore') as file: 
        test_desc = [line[1:-2] for line in file]

    with open("code-docstring-corpus-V2/repo_split/repo_split.parallel_desc.valid", 
            "r", encoding="utf-8", errors='ignore') as file: 
        valid_desc = [line[1:-2] for line in file]


    with open("code-docstring-corpus-V2/repo_split/repo_split.parallel_decl.train", 
            "r", encoding="utf-8", errors='ignore') as file: 
        train_decl = [line[:-1] for line in file]

    with open("code-docstring-corpus-V2/repo_split/repo_split.parallel_decl.test", 
            "r", encoding="utf-8", errors='ignore') as file: 
        test_decl = [line[:-1] for line in file]

    with open("code-docstring-corpus-V2/repo_split/repo_split.parallel_decl.valid", 
            "r", encoding="utf-8", errors='ignore') as file: 
        valid_decl = [line[:-1] for line in file]


    with open("code-docstring-corpus-V2/repo_split/repo_split.parallel_bodies.train", 
            "r", encoding="utf-8", errors='ignore') as file: 
        train_bodies = [line[1:-1] for line in file]

    with open("code-docstring-corpus-V2/repo_split/repo_split.parallel_bodies.test", 
            "r", encoding="utf-8", errors='ignore') as file: 
        test_bodies = [line[1:-1] for line in file]

    with open("code-docstring-corpus-V2/repo_split/repo_split.parallel_bodies.valid", 
            "r", encoding="utf-8", errors='ignore') as file: 
        valid_bodies = [line[1:-1] for line in file]

    return (train_desc, test_desc, valid_desc, 
            train_decl, test_decl, valid_decl, 
            train_bodies, test_bodies, valid_bodies)

if __name__ == "__main__":
    (tr_desc, tt_desc, vd_desc, 
     tr_decl, tt_decl, vd_decl, 
     tr_bodies, tt_bodies, vd_bodies) = data()
    print(f"tr_desc shape: {len(tr_desc)}")
    
