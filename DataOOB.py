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
    set_seed()
    DATASET_SIZE=100
    NUM_BOOTSTRAP=50
    NUM_DATA_IN_BOOTSTRAP=50
    train, dev, test=initialize_dataset(DATASET_SIZE)

    print(f"Initialized SST-2 with length of {len(train)}")
    # classifiers=[Bert_Classifier]
    classifiers=[LogReg_Classifier, RNN_Classifier, Bert_Classifier]
    # names=['BERT']
    names=['LogReg', 'RNN', 'BERT']
    for name, classifier_algo in zip(names, classifiers):
        values=[0 for i in range(DATASET_SIZE)]
        plt.figure()
        print(f"Running LOO with {name} classifier")

        for t in range(1, NUM_BOOTSTRAP+1, 1):
            train_iter = copy.deepcopy(train)
            boostrap = train_iter.bootstrap_data(NUM_DATA_IN_BOOTSTRAP)
            print(len(train_iter))
            print(boostrap)
            fds
            set_seed()
            Classifier = classifier_algo(train)
            results=Classifier.train_test(train, dev, test)
            dev_baseline=results[1]
            test_baseline=results[2]

            print(f"Results with all data is {results}")

            results=[]
            for i in range(DATASET_SIZE):
                train_loo=copy.deepcopy(train)
                train_loo.data.pop(i)
                set_seed()
                train_loo.reset_index()
                Classifier = classifier_algo(train_loo)
                _, dev_res, _ = Classifier.train_test(train_loo, dev, test)
                #If the perfo augments when removing (if diff is positive), then this was a bad data
                values[i] = ((t - 1) / t) * values[i] + (dev_res-dev_baseline) / t
                # values[i]=(dev_res-dev_baseline)
        # print(values)
        # fd
        results=values
        sorted_results=np.argsort(results)
        # print(sorted_results)
        # print(sorted_results[::-1])
        results_remove_best=[test_baseline]
        results_remove_worst=[test_baseline]
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
            results_remove_best.append(test_res)
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
            _, dev_res, _ = Classifier.train_test(train_eval, dev, test)
            results_remove_worst.append(test_res)
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
        plt.title(f'{name}-LOO-{NUM_ITER}-iters')
        plt.savefig(f'{name}LOO{NUM_ITER}iter.png')


if __name__ == '__main__':
    main()