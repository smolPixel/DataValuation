from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score
from data.DataProcessor import ds_DAControlled
import itertools
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np

class RNN_Classifier(pl.LightningModule):

    def __init__(self, argdict, train):
        super().__init__()
        self.num_directions = 2
        self.hidden_size = argdict['hidden_size_classifier']
        self.input_size = argdict['input_size']
        self.embedding = nn.Embedding(argdict['input_size'], argdict['embed_size_classifier'])
        self.bridge = nn.Linear(argdict['embed_size_classifier'], argdict['hidden_size_classifier'])
        # TODO: Test bidirectional
        self.rnn = nn.LSTM(argdict['embed_size_classifier'], argdict['hidden_size_classifier'],
                           argdict['num_layers_classifier'], batch_first=True, dropout=argdict['dropout_classifier'], bidirectional=True)
        self.dropout = nn.Dropout(argdict['dropout_classifier'])
        # if len(argdict['categories']) == 2:
        #     self.out = nn.Linear(self.hidden_size * 2, 1)
        # else:
        self.out = nn.Linear(self.hidden_size * 2, len(argdict['categories']))

        self.n_layers = argdict['num_layers_classifier']
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.argdict=argdict

    def init_model(self):
        self.linear_layer=nn.Linear(self.argdict['input_size'], len(self.argdict['categories']))
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # for param in self.model.base_model.parameters():
        #     param.requires_grad = False
        # self.optimizer = AdamW(self.model.parameters(), lr=1e-5)

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

    def training_step(self, batch, batch_idx):
        input=batch['input']
        bs = input.shape[0]
        output=self.forward(input)
        best=torch.softmax(output, dim=-1)
        pred=torch.argmax(best, dim=-1)
        acc=accuracy_score(batch['label'].cpu(), pred.cpu())
        loss=self.loss_function(output, batch['label'])
        self.log("Loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=False,
                 batch_size=bs)
        self.log("Acc Train", acc, on_epoch=True, on_step=False, prog_bar=True, logger=False,
                 batch_size=bs)
        return loss


    def validation_step(self, batch, batch_idx):
        input=batch['input']
        bs=input.shape[0]
        output=self.forward(input)
        best=torch.softmax(output, dim=-1)
        pred=torch.argmax(best, dim=-1)
        acc=accuracy_score(batch['label'].cpu(), pred.cpu())

        loss=self.loss_function(output, batch['label'])
        # self.log("Loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=False,
        #          batch_size=bs)
        self.log("Acc Dev", acc, on_epoch=True, on_step=False, prog_bar=True, logger=False,
                 batch_size=bs)
        return loss

    def test_step(self, batch, batch_idx):
        input=batch['input']
        bs=input.shape[0]
        output=self.forward(input)
        best=torch.softmax(output, dim=-1)
        pred=torch.argmax(best, dim=-1)
        acc=accuracy_score(batch['label'].cpu(), pred.cpu())
        self.acc_per_batch.append(acc)
        loss=self.loss_function(output, batch['label'])
        # self.log("Loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=False,
        #          batch_size=bs)
        self.log("Acc Dev", acc, on_epoch=True, on_step=False, prog_bar=True, logger=False,
                 batch_size=bs)
        return loss

    def validation_epoch_end(self, outputs):
        print("---\n")

    def train_model(self, training_set, dev_set, test_set, generator, return_grad=False):
        self.trainer = pl.Trainer(gpus=1, max_epochs=self.argdict['nb_epoch_classifier'], precision=16, enable_checkpointing=False)
        # trainer=pl.Trainer(max_epochs=self.argdict['num_epochs'])
        train_loader = DataLoader(
            dataset=training_set,
            batch_size=64,
            shuffle=True,
            # num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )
        dev_loader = DataLoader(
            dataset=dev_set,
            batch_size=64,
            shuffle=False,
            # num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )
        self.trainer.fit(self, train_loader, dev_loader)
        for set in [training_set, dev_set, test_set]:
            self.acc_per_batch=[]
            train_loader = DataLoader(
                dataset=set,
                batch_size=1,
                shuffle=True,
                # num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )
            train_acc=self.trainer.test(self, train_loader)
            acc=np.mean(self.acc_per_batch)
            print(self.acc_per_batch)
            print(acc)
            print(train_acc)
        fds
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

