import os, re, csv
import tqdm
from pathlib import Path
from dataclasses import dataclass

# code taken from
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(train, test):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    plt.plot(train, label="train loss")
    plt.plot(test, label="test loss")
    plt.xlabel("epochs")
    plt.ylabel("average negative log likelihood (NLL) loss")
    plt.savefig('smol_exp_losses.png')

seed = 10617
np.random.seed(seed)

MAX_LENGTH = 50
EPOCHS = 10

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = nn.functional.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


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
    n_tr, n_vd, n_tt = len(tr_desc), len(vd_desc), len(tt_desc)
    print(f"training examples: {n_tr}; validation examples: {n_vd}, test examples: {n_tt}\n")

    tr_x = [tr_decl[i] + " DCNL " + tr_bodies[i] for i in range(n_tr)]
    trainset = [[tr_x[i], tr_desc[i]] for i in range(n_tr)]
    # filter out really long examples

    def filterPair(p):
        return len(p[0].split(' ')) < MAX_LENGTH and \
                len(p[1].split(' ')) < MAX_LENGTH

    def filterPairs(pairs):
        return [pair for pair in pairs if filterPair(pair)]
        
    trainset = filterPairs(trainset)

    n_tr = len(trainset)
    print(f"new training examples: {n_tr}")

    tt_x = [tt_decl[i] + " DCNL " + tt_bodies[i] for i in range(n_tt)]
    testset = [[tt_x[i], tt_desc[i]] for i in range(n_tt)]
    testset = filterPairs(testset)
    n_tt = len(testset)
    print(f"new test examples: {n_tt}\n")

    vd_x = [vd_decl[i] + " DCNL " + vd_bodies[i] for i in range(n_vd)]
    validset = [[vd_x[i], vd_desc[i]] for i in range(n_vd)]


    input_lang = Lang("code")
    output_lang = Lang("desc")


    for pair in trainset:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    for pair in testset:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    def indexesFromSentence(lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(lang, sentence):
        indexes = indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

    def tensorsFromPair(pair):
        input_tensor = tensorFromSentence(input_lang, pair[0])
        target_tensor = tensorFromSentence(output_lang, pair[1])
        return (input_tensor, target_tensor)

    def get_dataloader_train(batch_size):
        n = len(trainset)
        input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
        target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

        for idx, (inp, tgt) in enumerate(trainset):
            inp_ids = indexesFromSentence(input_lang, inp)
            tgt_ids = indexesFromSentence(output_lang, tgt)
            inp_ids.append(EOS_token)
            tgt_ids.append(EOS_token)
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                                torch.LongTensor(target_ids).to(device))

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        return input_lang, output_lang, train_dataloader
    
    def get_dataloader_test(batch_size):
        n = len(testset)
        input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
        target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

        for idx, (inp, tgt) in enumerate(testset):
            inp_ids = indexesFromSentence(input_lang, inp)
            tgt_ids = indexesFromSentence(output_lang, tgt)
            inp_ids.append(EOS_token)
            tgt_ids.append(EOS_token)
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        test_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                                torch.LongTensor(target_ids).to(device))

        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
        return test_dataloader
    
    def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
        total_loss = 0
        for data in tqdm.tqdm(dataloader, desc="training epoch"):
            input_tensor, target_tensor = data

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
    

    def eval_epoch(dataloader, encoder, decoder, criterion):

        total_loss = 0
        for data in tqdm.tqdm(dataloader, desc="evaluating epoch"):
            input_tensor, target_tensor = data

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )

            total_loss += loss.item()

        return total_loss / len(dataloader)
    

    def train(train_dataloader, test_dataloader, encoder, decoder, n_epochs, learning_rate=0.001):
        train_losses = []
        test_losses = []

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        for _ in tqdm.tqdm(range(1, n_epochs + 1), desc="epoch"):
            train_losses.append(train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion))
            print('done training\n')
            test_losses.append(eval_epoch(test_dataloader, encoder, decoder, criterion))

            print(f"train: {train_losses[-1]}\n")
            print(f"test: {test_losses[-1]}\n")

        print(f"train losses: {train_losses}")
        print(f"test losses: {test_losses}")
        showPlot(train_losses, test_losses)


    def evaluate(encoder, decoder, sentence, input_lang, output_lang):
        with torch.no_grad():
            input_tensor = tensorFromSentence(input_lang, sentence)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, decoder_attn = decoder(encoder_outputs, encoder_hidden)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()

            decoded_words = []
            for idx in decoded_ids:
                if idx.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                decoded_words.append(output_lang.index2word[idx.item()])
        return decoded_words, decoder_attn
    
    def evaluateRandomly(encoder, decoder, n=10):
        for _ in range(n):
            pair = random.choice(trainset)
            print('>', pair[0])
            print('=', pair[1])
            output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

    hidden_size = 128
    batch_size = 128

    input_lang, output_lang, train_dataloader = get_dataloader_train(batch_size)
    test_dataloader = get_dataloader_test(batch_size)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    train(train_dataloader, test_dataloader, encoder, decoder, EPOCHS)

    encoder.eval()
    decoder.eval()
    evaluateRandomly(encoder, decoder)


        
