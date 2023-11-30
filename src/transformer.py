import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
