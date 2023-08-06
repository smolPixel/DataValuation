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

class Linear_Classifier(pl.LightningModule):

    def __init__(self, argdict, train):
        super().__init__()
        self.argdict=argdict
        self.vocab_size=argdict['input_size']


        self.embedding_size=300

        self.init_model()
        self.loss_function=train.loss_function
        # self.loss_function=
        # print(self.model)

    def init_model(self):
        if self.argdict['need_embedding']:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
            self.linear_layer=nn.Linear(self.embedding_size, len(self.argdict['categories']))
        else:
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
        if self.argdict['need_embedding']:
            input=self.embedding(input)
            input=torch.mean(input, dim=1)
        input_sequence = input.view(-1, self.argdict['input_size']).to('cuda').float()
        output=self.linear_layer(input_sequence)
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
        if self.argdict['need_embedding']:
            input=self.embedding(input)
            input=torch.mean(input, dim=1)
        input_sequence = input.view(-1, self.argdict['input_size']).to('cuda').float()
        output=self.linear_layer(input_sequence)
        best=torch.softmax(output, dim=-1)
        pred=torch.argmax(best, dim=-1)
        acc=accuracy_score(batch['label'].cpu(), pred.cpu())

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
        # fds


    def forward(self, inputs):
        l1=self.embedding(inputs)
        outputs=self.linear_layer(l1)
        return outputs

