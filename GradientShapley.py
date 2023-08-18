"""Calculating values with Gradient Shapley, from https://arxiv.org/pdf/1904.02868.pdf"""


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

from Classifier.LogReg import LogReg_Classifier
from Classifier.RNN import RNN_Classifier
from Classifier.BERT import Bert_Classifier
from torch.utils.data import DataLoader
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def Gradient_Shapley(train, dev, test, classifier_algo, dev_baseline):
    #Let's program one iteration. First we need to randomly permute data point
    #dev_baseline= V(D) in paper

    #For now let's put PT at 2%, aka, when we get at 2% of the value of dev_baseline we are satisfied
    PT=0.02
    #TODO FINE TUNING WITH ONE EPOCH

    #baseline is 0.5 for binary dataset
    vals=[0.5 for i in range(len(train))]
    phis=[0 for i in range(len(train))]
    phis_prec=phis
    t=1
    for t in tqdm(range(1, 10, 1)):
        train_iter = copy.deepcopy(train)
        permuatation = train_iter.permute_data()
        set_seed()
        # print(len(train_trunc))
        # 1e-6: 50.9 dont learn
        # 1e-5:49.1
        # 1e-4: 51.1
        # 1e-3 : 51.0
        # 1e-2:50.9
        Classifier = classifier_algo(train_iter, lr=1e-3)
        train_loader = DataLoader(
            dataset=train_iter,
            batch_size=1,
            shuffle=False,
            # num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )
        for j, batch in enumerate(train_loader):
            Classifier.optimizer.zero_grad()
            loss=Classifier.forward(batch)
            loss.backward()
            Classifier.optimizer.step()
            vjt=Classifier.evaluate(dev)
            new_point=permuatation[j]
            phis[new_point] = ((t - 1) / t) * phis[new_point] + (vals[j - 1] - vjt) / t
        #     fds
        #     new_point=train_iter.data[j]
        #     print(new_point)
        #     print(dat)
        #     fds
        #
        #     _, vjt, _ = Classifier.train_test(train_trunc, dev, test)
        #     vals[j]=vjt
        #         #Inverse to the paper, to keep in track with loo: baseline is with the point included
        #     phis[new_point]=((t-1)/t)*phis[new_point]+(vals[j-1]-vjt)/t
        # phis_prec=phis

    return phis

def main():
    set_seed()
    DATASET_SIZE=100
    train, dev, test=initialize_dataset(DATASET_SIZE)

    print(f"Initialized SST-2 with length of {len(train)}")
    # classifiers=[Bert_Classifier]
    classifiers=[RNN_Classifier, Bert_Classifier]
    # names=['BERT']
    names=['RNN', 'BERT']
    for name, classifier_algo in zip(names, classifiers):
        plt.figure()
        print(f"Running LOO with {name} classifier")

        set_seed()
        Classifier = classifier_algo(train)
        results=Classifier.train_test(train, dev, test)
        dev_baseline=results[1]
        test_baseline=results[2]

        print(f"Results with all data is {results}")

        results=[]

        shapleys=Gradient_Shapley(train, dev, test, classifier_algo, dev_baseline)
        fds
        sorted_results=np.argsort(shapleys)
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
            _, _, test_res = Classifier.train_test(train_eval, dev, test)
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
            _, _, test_res = Classifier.train_test(train_eval, dev, test)
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
        plt.title(f'{name}-Gradient')
        plt.savefig(f'{name}Gradient.png')
        print("before continuing, you need finetuning bert")
        fds


if __name__ == '__main__':
    main()