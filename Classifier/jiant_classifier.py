from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import os
from sklearn.metrics import accuracy_score
import Classifier.jiant.proj.simple.runscript as simple_run
import shutil

class Jiant_Classifier():

    def __init__(self, argdict):
        self.argdict=argdict
        if argdict['dataset'].lower() in ['sst-2', "fakenews", "irony"]:
            self.task='SST2'
        elif argdict['dataset'].lower() in ["trec6"]:
            self.task="TREC6"
        self.MODEL_TYPE = "bert-base-uncased"

        self.RUN_NAME = f"/Tmp/piedboef/simple_{self.task}_{self.MODEL_TYPE}"
        self.EXP_DIR = f"/Tmp/piedboef/content/exp"
        #Data dir needs to be a global path, probably due to the
        self.DATA_DIR = f"/Tmp/piedboef/data/"

        # print(self.model)

        shutil.rmtree(self.DATA_DIR, ignore_errors=True)
        shutil.rmtree(self.EXP_DIR, ignore_errors=True)
        shutil.rmtree(self.RUN_NAME, ignore_errors=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.EXP_DIR, exist_ok=True)
        os.makedirs(f"{self.DATA_DIR}/configs", exist_ok=True)
        #Binary tasks are mapped to SST-2
        if argdict['dataset'].lower() in ['sst-2', "fakenews", "irony"]:
            os.makedirs(f"{self.DATA_DIR}/sst", exist_ok=True)
            self.task_name="sst"
        elif argdict['dataset'].lower() in ["trec6"]:
            os.makedirs(f"{self.DATA_DIR}/trec", exist_ok=True)
            self.task_name="trec"
        else:
            raise ValueError("Task not found")



    def train(self, train, dev, type='dataLoader', generator=None, return_grad=False):

        train_df=train.return_pandas()
        dev_df=dev.return_pandas()
        train_df.to_csv(f"/Tmp/piedboef/data/{self.task_name}/train.tsv", sep='\t')
        dev_df.to_csv(f"/Tmp/piedboef/data/{self.task_name}/dev.tsv", sep='\t')
        # dev_df.to_tsv("/Tmp/piedboef/data/bert/data/sst/tr.tsv", index_col=0)
        if self.argdict['dataset'].lower() in ['sst-2', "fakenews", "irony"]:
            file='{"task": "SST2","paths": {"train": "/Tmp/piedboef/data/sst/train.tsv","test": "/Tmp/piedboef/data/sst/dev.tsv",' \
                 '"val": "/Tmp/piedboef/data/sst/dev.tsv"}, "name": "SST2"}'
            textfile=open("/Tmp/piedboef/data/configs/SST2_config.json", "w")
        elif self.argdict['dataset'].lower() in ["trec6"]:
            file = '{"task": "TREC6","paths": {"train": "/Tmp/piedboef/data/trec/train.tsv","test": "/Tmp/piedboef/data/trec/dev.tsv",' \
                   '"val": "/Tmp/piedboef/data/trec/dev.tsv"}, "name": "TREC6"}'
            textfile = open("/Tmp/piedboef/data/configs/TREC6_config.json", "w")
        a=textfile.write(file)
        textfile.close()
        args = simple_run.RunConfiguration(
            run_name=self.RUN_NAME,
            exp_dir=self.EXP_DIR,
            data_dir=self.DATA_DIR,
            model_type=self.MODEL_TYPE,
            train_tasks=self.task,
            val_tasks=self.task,
            test_tasks=self.task,
            train_batch_size=16,
            num_train_epochs=self.argdict['nb_epoch_classifier'],
            seed=self.argdict['random_seed']
        )
        train_acc, dev_acc=simple_run.run_simple(args)
        # print(results['SST2']['metrics'].major)
        if self.argdict['dataset'].lower() in ['sst-2', "fakenews", "irony"]:
            return train_acc['SST2']['metrics'].major, dev_acc['SST2']['metrics'].major
        elif self.argdict['dataset'].lower() in ["trec6"]:
            # print(dev_acc)
            # fds
            return train_acc['TREC6']['metrics'].minor['acc'], dev_acc['TREC6']['metrics'].minor['acc']

