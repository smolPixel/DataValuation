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
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
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
    GRADIENT_ACCUMULATION_STEP=10
    C=1e7
    R=10
    BS=10
    NUM_SAMPLES=8000
    damping=0.01
    weight_decay=0.01
    train_sampler = RandomSampler(train_dataset,
                                  replacement=True,
                                  num_samples=NUM_SAMPLES)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=BS // GRADIENT_ACCUMULATION_STEP)


    no_decay = ['bias', 'LayerNorm.weight']
    final_res = None
    for r in range(R):
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
                                            algo.model.parameters(),
                                            create_graph=True)
            grad = []
            H = 0
            for i, (g, g_v) in enumerate(zip(grad_list, res)):
                H += (g * g_v).sum() / GRADIENT_ACCUMULATION_STEP
            # H = grad @ v
            H.backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEP == 0:

                # print(res[20])

                for i, ((n, p),
                        v_p) in enumerate(zip(algo.model.named_parameters(), res)):
                    try:
                        if not any(nd in n for nd in no_decay):
                            res[i] = (1 - damping) * v_p - (
                                p.grad.data.add_(weight_decay,
                                                 v_p)) / C + grads[i].cuda()
                        else:
                            res[i] = (1 - damping) * v_p - (
                                p.grad.data) / C + grads[i].cuda()
                    except RuntimeError:

                        v_p = v_p.cpu()

                        p_grad = p.grad.data.cpu()

                        if not any(nd in n for nd in no_decay):
                            res[i] = ((1 - damping) * v_p -
                                      (p_grad.add_(args.weight_decay, v_p)) /
                                      C + v[i]).cuda()
                        else:
                            res[i] = ((1 - damping) * v_p -
                                      (p_grad) / C + v[i]).cuda()
                algo.model.zero_grad()

            if final_res is None:
                final_res = [(b / C).cpu().float() for b in res]
            else:
                final_res = [
                    a + (b / C).cpu().float() for a, b in zip(final_res, res)
                ]

    final_res = [a / float(R) for a in final_res]
    return final_res

def get_influence(training_set, algo, HVP):
    eval_sampler = SequentialSampler(training_set)
    eval_dataloader = DataLoader(training_set,
                                 sampler=eval_sampler,
                                 batch_size=1)
    HVP = [it.cuda() for it in HVP]
    no_decay = ['bias', 'LayerNorm.weight']
    count=0
    WEIGHT_DECAY=0.01
    negative_count = 0
    influence_list = []
    for batch in tqdm(eval_dataloader, desc="Calculating validation grad"):
        # if count > 10:
        #    break
        algo.model.eval()
        # batch = tuple(t.to(args.device) for t in batch)
        text_batch = batch['sentence']
        encoding = algo.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].cuda()
        attention_mask = encoding['attention_mask'].cuda()
        # print(encoding)
        labels = batch['label'].cuda()
        outputs = algo.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        count += 1
        influence = 0
        for i, ((n, p), v) in enumerate(zip(algo.model.named_parameters(), HVP)):
            if p.grad is None:
                print("wrong")
            else:
                if not any(nd in n for nd in no_decay):
                    influence += ((p.grad.data.add_(WEIGHT_DECAY, p.data))*v).sum() * -1
                #                    influence += ((p.grad.data)*v).sum() * -1
                else:
                    influence += ((p.grad.data) * v).sum() * -1

        if influence.item() < 0:
            negative_count += 1
        influence_list.append(influence.item())
        if count % 100 == 0:
            print(influence.item())
            print(negative_count / count)
    influence_list = np.array(influence_list)
    return influence_list


def InfluenceFunction(train, dev, test, classifier_algo):
    #Calculating the values of each point in the training set with influence function
    # Step 1: Get the HPV
    R=10

    #First, get validation gradient
    grad=get_validation_grad(classifier_algo, dev)
    HPV=get_HPV(train, classifier_algo, grad)
    return get_influence(train, classifier_algo, HPV)




    return phis

def main():
    split_test = 'test'
    DATASET_SIZE=100
    train, dev, test=initialize_dataset(DATASET_SIZE)

    print(f"Initialized SST-2 with length of {len(train)}")
    # classifiers=[Bert_Classifier]
    classifiers=[Bert_Classifier]
    lrs=[1e-5]
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

        Classifier = classifier_algo(train)
        results=Classifier.train_test(train, dev, test)
        dev_baseline=results[1]
        test_baseline=results[2]

        print(f"Results with all data is {results}")

        results=[]

        influenceFunc=InfluenceFunction(train, dev, test, Classifier)
        sorted_results=np.argsort(influenceFunc)
        # print(sorted_results)
        # print(sorted_results[::-1])
        if split_test == 'dev':
            baseline = dev_baseline
        else:
            baseline = test_baseline
        results_remove_best = [baseline]
        results_remove_worst = [baseline]
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
            _, dev_res, test_res = Classifier.train_test(train_eval, dev, test)
            if split_test == 'test':
                results_remove_best.append(test_res)
            else:
                results_remove_best.append(dev_res)
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
            _, dev_res, test_res = Classifier.train_test(train_eval, dev, test)
            if split_test == 'test':
                results_remove_worst.append(test_res)
            else:
                results_remove_worst.append(dev_res)
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
        plt.title(f'{name}-Influence')
        plt.savefig(f'{name}Influence.png')


if __name__ == '__main__':
    main()