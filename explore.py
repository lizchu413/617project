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
print(f"torch cuda available: {torch.cuda.is_available()}")

seed = 10617
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)