from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score
import itertools

import numpy as np


class RNN_Classifier(nn.Module):

    def __init__(self,train):
        super(RNN_Classifier, self).__init__()
        self.num_directions = 2
        self.hidden_size = 1024#argdict['hidden_size_classifier']
        self.input_size = train.vocab_size #argdict['input_size']
        self.embedding = nn.Embedding(self.input_size, 300)#argdict['embed_size_classifier'])
        self.bridge = nn.Linear(300, 1024)#argdict['hidden_size_classifier'])
        # TODO: Test bidirectional
        self.rnn = nn.LSTM(300, 1024, 2, batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        # if len(argdict['categories']) == 2:
        #     self.out = nn.Linear(self.hidden_size * 2, 1)
        # else:
        self.out = nn.Linear(self.hidden_size * 2, 2)

        self.n_layers = 2
        self.loss_function = torch.nn.CrossEntropyLoss()

    def init_model(self):
        self.linear_layer=nn.Linear(self.argdict['input_size'], len(self.argdict['categories']))
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # for param in self.model.base_model.parameters():
        #     param.requires_grad = False
        # self.optimizer = AdamW(self.model.parameters(), lr=1e-5)

    def train_test(self, train, dev, test):
        for ep in range(10):
            train_loader = DataLoader(
                dataset=train,
                batch_size=25,
                shuffle=True,
                # num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )
            for batch in train_loader:
                self.forward(batch['input'])
                fds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def get_logits(self, batch):
        input = batch['input']
        bs = input.shape[0]
        embed = self.embedding(input)
        embed = torch.mean(embed, dim=1)
        output = self.linear_layer(embed)
        return output

    def get_loss(self, batch):
        input=batch['input']
        bs = input.shape[0]
        embed=self.embedding(input)
        embed=torch.mean(embed, dim=1)
        output=self.linear_layer(embed)
        best=torch.softmax(output, dim=-1)
        pred=torch.argmax(best, dim=-1)
        acc=accuracy_score(batch['label'].cpu(), pred.cpu())
        loss=self.loss_function(output, batch['label'])
        return loss

        # fds


    def forward(self, input):
        input=input
        bs=input.shape[0]
        input=input.cuda()
        input = self.embedding(input)
        input = self.dropout(input)
        # input = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        input, (hidden, cell_state) = self.rnn(input)
        #Keep only last state
        # input, _= nn.utils.rnn.pad_packed_sequence(input, batch_first=True)
        # seq_len=input.shape[1]
        # input=input.contiguous().view(seq_len, bs, self.num_directions*self.hidden_size)
        # input=input[-1]
        hidden=hidden.view(self.n_layers, self.num_directions, -1, self.hidden_size)
        hidden=torch.cat([hidden[-1, -1], hidden[-1, -2]], dim=1)
        # Getting the output over vocabulary
        output = self.out(hidden)
        return output

