"""Baseline with LOO experiments"""


#Main file
from data.DataProcessor import initialize_dataset
from Classifier.classifier import classifier
import argparse
import random
import numpy as np
import torch
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

from Classifier.LogReg import LogReg_Classifier
from Classifier.RNN import RNN_Classifier
from Classifier.BERT import Bert_Classifier
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    split_test = 'dev'
    DATASET_SIZE=100
    NUM_BOOTSTRAP=500
    NUM_DATA_IN_BOOTSTRAP=25
    train, dev, test=initialize_dataset(DATASET_SIZE)

    print(f"Initialized SST-2 with length of {len(train)}")
    # classifiers=[Bert_Classifier]
    classifiers=[Bert_Classifier]
    # names=['BERT']
    names=['BERT']
    for name, classifier_algo in zip(names, classifiers):
        values=[[] for i in range(DATASET_SIZE)]
        plt.figure()
        print(f"Running LOO with {name} classifier")
        Classifier = classifier_algo(train)
        results = Classifier.train_test(train, dev, test)
        dev_baseline = results[1]
        test_baseline = results[2]

        for t in range(1, NUM_BOOTSTRAP+1, 1):
            train_iter = copy.deepcopy(train)
            boostrap = train_iter.bootstrap_data(NUM_DATA_IN_BOOTSTRAP)
            Classifier = classifier_algo(train_iter)
            results=Classifier.train_test(train_iter, dev, test)
            dev_results=results[1]


            for i in range(DATASET_SIZE):
                if i not in boostrap:
                    values[i].append(dev_results)

        values=[np.mean(vv) for vv in values]
        results=values
        sorted_results=np.argsort(results)
        # print(sorted_results)
        # print(sorted_results[::-1])
        if split_test == 'dev':
            baseline = dev_baseline
        else:
            baseline = test_baseline
        results_remove_best = [baseline]
        results_remove_worst = [baseline]
        print("Evaluation of OOB, removing best data by bs of 10")
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
        plt.title(f'{name}-DataOOB')
        plt.savefig(f'{name}DataOOB.png')


if __name__ == '__main__':
    main()