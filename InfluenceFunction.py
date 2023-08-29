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


def InfluenceFunction(train, dev, test, classifier_algo):
    #Calculating the values of each point in the training set with influence function


    return phis

def main():
    set_seed()
    DATASET_SIZE=100
    train, dev, test=initialize_dataset(DATASET_SIZE)

    print(f"Initialized SST-2 with length of {len(train)}")
    # classifiers=[Bert_Classifier]
    classifiers=[RNN_Classifier, Bert_Classifier]
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

        shapleys=Gradient_Shapley(train, dev, test, classifier_algo, dev_baseline, lr)
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