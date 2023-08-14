"""Calculating values with TMC Shapley, from https://arxiv.org/pdf/1904.02868.pdf"""


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


def TMC_Shapley(train, dev, test, algo):
    #Let's program one iteration. First we need to randomly permute data point
    train_iter=copy.deepcopy(train)
    train_iter.permute_data()
    fds

def main():
    set_seed()
    DATASET_SIZE=100
    train, dev, test=initialize_dataset(DATASET_SIZE)

    print(f"Initialized SST-2 with length of {len(train)}")
    # classifiers=[Bert_Classifier]
    classifiers=[LogReg_Classifier, RNN_Classifier, Bert_Classifier]
    # names=['BERT']
    names=['LogReg', 'RNN', 'BERT']
    for name, classifier_algo in zip(names, classifiers):
        plt.figure()
        print(f"Running LOO with {name} classifier")

        set_seed()
        Classifier = classifier_algo(train)
        results=Classifier.train_test(train, dev, test)
        dev_baseline=results[1]

        print(f"Results with all data is {results}")

        results=[]

        shapleys=TMC_Shapley(train, dev, test, classifier_algo)

        for i in range(DATASET_SIZE):
            train_loo=copy.deepcopy(train)
            train_loo.data.pop(i)
            set_seed()
            train_loo.reset_index()
            Classifier = classifier_algo(train_loo)
            _, dev_res, _ = Classifier.train_test(train_loo, dev, test)
            #If the perfo augments when removing (if diff is positive), then this was a bad data
            results.append(dev_res-dev_baseline)

        sorted_results=np.argsort(results)
        # print(sorted_results)
        # print(sorted_results[::-1])
        results_remove_best=[dev_baseline]
        results_remove_worst=[dev_baseline]
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
            _, dev_res, _ = Classifier.train_test(train_eval, dev, test)
            results_remove_best.append(dev_res)
            print(f"Results of {dev_res}")
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
            results_remove_worst.append(dev_res)
            print(f"Results of {dev_res}")


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
        plt.title(f'{name}-LOO')
        plt.savefig(f'{name}LOO.png')
    fds
    # print("Running LOO with RNN classifier")
    #
    # set_seed()
    # Classifier = RNN_Classifier(train)
    # results = Classifier.train_test(train, dev, test)
    # print(f"Results with all data is {results}")
    # dev_baseline = results[1]
    # results = []
    # results_remove_best=[dev_baseline]
    # results_remove_worst=[dev_baseline]
    # for i in tqdm(range(DATASET_SIZE)):
    #     train_loo=copy.deepcopy(train)
    #     train_loo.data.pop(i)
    #     train_loo.reset_index()
    #     set_seed()
    #     Classifier = RNN_Classifier(train_loo)
    #     _, dev_res, _ = Classifier.train_test(train_loo, dev, test)
    #     #If the perfo augments when removing (if diff is positive), then this was a bad data
    #     results.append(dev_res-dev_baseline)
    #
    # sorted_results=np.argsort(results)
    # # print(sorted_results)
    # # print(sorted_results[::-1])
    # print("Evaluation of LOO, removing best data by bs of 10")
    # # for i in range(0, 55, 5):
    # for i in range(1, 5, 1):
    #     train_eval=copy.deepcopy(train)
    #     eliminated=0
    #     for ss in sorted_results:
    #         if eliminated == i:
    #             print(f"Testing with dataset of size of {len(train_eval)}")
    #             break
    #         train_eval.data.pop(ss)
    #         eliminated+=1
    #     set_seed()
    #     Classifier = RNN_Classifier(train_eval)
    #     train_eval.reset_index()
    #     _, dev_res, _ = Classifier.train_test(train_eval, dev, test)
    #     print(f"Results of {dev_res}")
    #     results_remove_best.append(dev_res)
    # print("Evaluation of LOO, removing worst data by bs of 10")
    # # for i in range(0, 55, 5):
    # for i in range(0, 5, 1):
    #     train_eval = copy.deepcopy(train)
    #     eliminated = 0
    #     for ss in sorted_results[::-1]:
    #         if eliminated == i:
    #             print(f"Testing with dataset of size of {len(train_eval)}")
    #             break
    #         train_eval.data.pop(ss)
    #         eliminated += 1
    #     set_seed()
    #     Classifier = RNN_Classifier(train_eval)
    #     train_eval.reset_index()
    #     _, dev_res, _ = Classifier.train_test(train_eval, dev, test)
    #     print(f"Results of {dev_res}")
    #     results_remove_worst.append(dev_res)
    #
    # X=[i for i in range(0,5,1)]
    # # X=[i for i in range(0,55,5)]
    # # X.extend([i for i in range(0,55,5)])
    # X.extend([i for i in range(0,5, 1)])
    # strats=['remove bad' for i in range(0,5,1)]
    # # strats=['remove bad' for i in range(0,55,5)]
    # strats.extend(['remove good' for i in range(0,5,1)])
    # # strats.extend(['remove good' for i in range(0,55,5)])
    # Y=results_remove_worst
    # Y.extend(results_remove_best)
    # data_plot=pd.DataFrame({'Number of data points removed': X, 'Accuracy': Y, 'Strategy':strats})
    # sns.lineplot(x='Number of data points removed', y='Accuracy', hue='Strategy', data=data_plot)
    # plt.title('RNN-LOO')
    # plt.savefig('RNNLOO.png')
    #
    # fds
    # print("Running LOO with BERT")
    #
    # set_seed()
    # Classifier = Bert_Classifier(train)
    # results = Classifier.train_test(train, dev, test)
    # print(f"Results with all data is {results}")
    # dev_baseline = results[1]
    # results = []
    # results_remove_best=[dev_baseline]
    # results_remove_worst=[dev_baseline]
    # for i in tqdm(range(DATASET_SIZE)):
    #     train_loo=copy.deepcopy(train)
    #     train_loo.data.pop(i)
    #     train_loo.reset_index()
    #     set_seed()
    #     Classifier = Bert_Classifier(train_loo)
    #     _, dev_res, _ = Classifier.train_test(train_loo, dev, test)
    #     #If the perfo augments when removing (if diff is positive), then this was a bad data
    #     results.append(dev_res-dev_baseline)
    #
    # sorted_results=np.argsort(results)
    # # print(sorted_results)
    # # print(sorted_results[::-1])
    # print("Evaluation of LOO, removing best data by bs of 10")
    # for i in range(0, 55, 5):
    #     train_eval=copy.deepcopy(train)
    #     eliminated=0
    #     for ss in sorted_results:
    #         if eliminated == i:
    #             print(f"Testing with dataset of size of {len(train_eval)}")
    #             break
    #         train_eval.data.pop(ss)
    #         eliminated+=1
    #     set_seed()
    #     Classifier = Bert_Classifier(train_eval)
    #     train_eval.reset_index()
    #     _, dev_res, _ = Classifier.train_test(train_eval, dev, test)
    #     print(f"Results of {dev_res}")
    #     results_remove_best.append(dev_res)
    # print("Evaluation of LOO, removing worst data by bs of 10")
    # for i in range(0, 55, 5):
    #     train_eval = copy.deepcopy(train)
    #     eliminated = 0
    #     for ss in sorted_results[::-1]:
    #         if eliminated == i:
    #             print(f"Testing with dataset of size of {len(train_eval)}")
    #             break
    #         train_eval.data.pop(ss)
    #         eliminated += 1
    #     set_seed()
    #     Classifier = Bert_Classifier(train_eval)
    #     train_eval.reset_index()
    #     _, dev_res, _ = Classifier.train_test(train_eval, dev, test)
    #     print(f"Results of {dev_res}")
    #     results_remove_worst.append(dev_res)
    # # results_train_iter, results_dev_iter, results_test_iter =classifier_algo.train_test(train, dev, test)
    #
    # X=[i for i in range(0,55,5)]
    # X.extend([i for i in range(0,55,5)])
    # strats=['remove bad' for i in range(0,55,5)]
    # strats.extend(['remove good' for i in range(0,55,5)])
    # Y=results_remove_worst
    # Y.extend(results_remove_best)
    # data_plot=pd.DataFrame({'Number of data points removed': X, 'Accuracy': Y, 'Strategy':strats})
    # sns.lineplot(x='Number of data points removed', y='Accuracy', hue='Strategy', data=data_plot)
    # plt.title('BERT-LOO')
    # plt.savefig('BERTLOO.png')

#66.74

if __name__ == '__main__':
    main()