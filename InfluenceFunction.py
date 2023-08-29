"""Calculating values with Influence Function, following https://arxiv.org/pdf/2004.11546.pdf"""


#Main file
from data.DataProcessor import initialize_dataset
from Classifier.classifier import classifier
import argparse
import random
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import yaml
import time
from datetime import timedelta
import copy
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc

from Classifier.LogReg import LogReg_Classifier
from Classifier.RNN import RNN_Classifier
from Classifier.BERT import Bert_Classifier
from torch.utils.data import DataLoader
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_validation_grad(model, dev):
    algo=model
    algo.model.eval()
    algo.optimizer.zero_grad()
    dev_loader = DataLoader(
        dataset=dev,
        batch_size=16,
        shuffle=False,
        num_workers=4,  # cpu_count(),
        pin_memory=torch.cuda.is_available()
    )
    for batch in tqdm(dev_loader, desc="Calculating validation grad"):
        #We basically accumulate grad over the whole validation dataset
        #if count > 10:
        #    break
        text_batch = batch['sentence']
        encoding = algo.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].cuda()
        attention_mask = encoding['attention_mask'].cuda()
        # print(encoding)
        labels = batch['label'].cuda()
        outputs = algo.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        grad = []
        for p in algo.model.parameters():
            if p.grad is None:
                print("wrong")
            #print(len(eval_dataset))
            grad.append((p.grad.data / len(dev)).cpu())

        #print(grad)
        return grad

def get_HPV(train_dataset, algo, grads):
    train_sampler = RandomSampler(train_dataset,
                                  replacement=True,
                                  num_samples=8000)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=10 // 10)

    no_decay = ['bias', 'LayerNorm.weight']
    final_res = None
    for r in trange(10):
        res = [w.clone().cuda() for w in grads]
        algo.optimizer.zero_grad()
        for step, batch in enumerate(
                tqdm(train_dataloader, desc="Calculating HVP")):
            algo.model.eval()
            text_batch = batch['sentence']
            encoding = algo.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].cuda()
            attention_mask = encoding['attention_mask'].cuda()
            # print(encoding)
            labels = batch['label'].cuda()
            outputs = algo.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            grad_list = torch.autograd.grad(loss,
                                            model.parameters(),
                                            create_graph=True)
            grad = []
            H = 0
            for i, (g, g_v) in enumerate(zip(grad_list, res)):
                H += (g * g_v).sum() / args.gradient_accumulation_steps
            # H = grad @ v
            H.backward()
            print(H)
            fds

def InfluenceFunction(train, dev, test, classifier_algo):
    #Calculating the values of each point in the training set with influence function
    # Step 1: Get the HPV
    R=10

    #First, get validation gradient
    grad=get_validation_grad(classifier_algo, dev)
    HPV=get_HPV(train, classifier_algo, grad)
    print(HPV)
    fds




    return phis

def main():
    set_seed()
    DATASET_SIZE=100
    train, dev, test=initialize_dataset(DATASET_SIZE)

    print(f"Initialized SST-2 with length of {len(train)}")
    # classifiers=[Bert_Classifier]
    classifiers=[Bert_Classifier]
    lrs=[1e-3, 1e-5]
    # Bert 1e-5 : 61.3 but overfits
    # 1e-6: 50.9
    # 1e-4: 50.9
    # 1e-3 :50.9
    # 1e-2

    # names=['BERT']
    names=['BERT']
    for name, classifier_algo, lr in zip(names, classifiers, lrs):
        if name=="RNN":
            continue
        plt.figure()
        print(f"Running LOO with {name} classifier")

        set_seed()
        Classifier = classifier_algo(train)
        results=Classifier.train_test(train, dev, test)
        dev_baseline=results[1]
        test_baseline=results[2]

        print(f"Results with all data is {results}")

        results=[]

        influenceFunc=InfluenceFunction(train, dev, test, Classifier)
        sorted_results=np.argsort(shapleys)
        # print(sorted_results)
        # print(sorted_results[::-1])
        results_remove_best=[test_baseline]
        results_remove_worst=[test_baseline]
        values_x = [0]
        values_x.extend([i for i in range(5, 55, 5)])
        print("Evaluation of LOO, removing best data by bs of 10")
        for i in range(5, 55, 5):
            train_eval=copy.deepcopy(train)
            eliminated=0
            for ss in sorted_results:
                if eliminated == i:
                    print(f"Testing with dataset of size of {len(train_eval)}")
                    break
                train_eval.data.pop(ss)
                eliminated+=1
            set_seed()
            train_eval.reset_index()
            Classifier = classifier_algo(train_eval)
            _, _, test_res = Classifier.train_test(train_eval, dev, test)
            results_remove_best.append(test_res)
            print(f"Results of {test_res}")
        auc_best = auc(values_x, results_remove_best)
        print(f"Area under curve is {auc_best}")
        print("Evaluation of LOO, removing worst data by bs of 10")
        for i in range(5, 55, 5):
            train_eval = copy.deepcopy(train)
            eliminated = 0
            for ss in sorted_results[::-1]:
                if eliminated == i:
                    print(f"Testing with dataset of size of {len(train_eval)}")
                    break
                train_eval.data.pop(ss)
                eliminated += 1
            set_seed()
            train_eval.reset_index()
            Classifier = classifier_algo(train_eval)
            _, _, test_res = Classifier.train_test(train_eval, dev, test)
            results_remove_worst.append(test_res)
            print(f"Results of {test_res}")
        auc_worst = auc(values_x, results_remove_worst)
        print(f"Area under curve is {auc_worst}")

        print(f"Final metric: {auc_worst - auc_best}")

        X=[i for i in range(0,55,5)]
        X.extend([i for i in range(0,55,5)])
        strats=['remove bad' for i in range(0,55,5)]
        strats.extend(['remove good' for i in range(0,55,5)])
        Y=results_remove_worst
        Y.extend(results_remove_best)
        print(X)
        print(Y)
        data_plot=pd.DataFrame({'Number of data points removed': X, 'Accuracy': Y, 'Strategy':strats})
        sns.lineplot(x='Number of data points removed', y='Accuracy', hue='Strategy', data=data_plot)
        plt.title(f'{name}-Gradient')
        plt.savefig(f'{name}Gradient.png')


if __name__ == '__main__':
    main()