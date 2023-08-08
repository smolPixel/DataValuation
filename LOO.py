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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed()
    DATASET_SIZE=100
    train, dev, test=initialize_dataset(DATASET_SIZE)
    print(f"Initialized SST-2 with length of {len(train)}")

    # print("Running LOO with LogReg classifier")
    #
    # from Classifier.LogReg import LogReg_Classifier
    # set_seed()
    # Classifier = LogReg_Classifier(train)
    # results=Classifier.train_test(train, dev, test)
    # dev_baseline=results[1]
    #
    # print(f"Results with all data is {results}")
    #
    # results=[]
    # for i in range(DATASET_SIZE):
    #     train_loo=copy.deepcopy(train)
    #     train_loo.data.pop(i)
    #     set_seed()
    #     Classifier = LogReg_Classifier(train_loo)
    #     _, dev_res, _ = Classifier.train_test(train_loo, dev, test)
    #     #If the perfo augments when removing (if diff is positive), then this was a bad data
    #     results.append(dev_res-dev_baseline)
    #
    # sorted_results=np.argsort(results)
    # # print(sorted_results)
    # # print(sorted_results[::-1])
    # print("Evaluation of LOO, removing best data by bs of 10")
    # for i in range(0, 50, 5):
    #     train_eval=copy.deepcopy(train)
    #     eliminated=0
    #     for ss in sorted_results:
    #         if eliminated == i:
    #             print(f"Testing with dataset of size of {len(train_eval)}")
    #             break
    #         train_eval.data.pop(ss)
    #         eliminated+=1
    #     set_seed()
    #     Classifier = LogReg_Classifier(train_eval)
    #     _, dev_res, _ = Classifier.train_test(train_eval, dev, test)
    #     print(f"Results of {dev_res}")
    # print("Evaluation of LOO, removing worst data by bs of 10")
    # for i in range(0, 50, 5):
    #     train_eval = copy.deepcopy(train)
    #     eliminated = 0
    #     for ss in sorted_results[::-1]:
    #         if eliminated == i:
    #             print(f"Testing with dataset of size of {len(train_eval)}")
    #             break
    #         train_eval.data.pop(ss)
    #         eliminated += 1
    #     set_seed()
    #     Classifier = LogReg_Classifier(train_eval)
    #     _, dev_res, _ = Classifier.train_test(train_eval, dev, test)
    #     print(f"Results of {dev_res}")

    print("Running LOO with RNN classifier")

    from Classifier.RNN import RNN_Classifier
    set_seed()
    Classifier = RNN_Classifier(train)
    results = Classifier.train_test(train, dev, test)
    dev_baseline = results[1]
    results = []
    print(f"Results with all data is {results}")
    for i in tqdm(range(DATASET_SIZE)):
        train_loo=copy.deepcopy(train)
        train_loo.data.pop(i)
        train_loo.reset_index()
        set_seed()
        Classifier = RNN_Classifier(train_loo)
        _, dev_res, _ = Classifier.train_test(train_loo, dev, test)
        #If the perfo augments when removing (if diff is positive), then this was a bad data
        results.append(dev_res-dev_baseline)

    sorted_results=np.argsort(results)
    # print(sorted_results)
    # print(sorted_results[::-1])
    print("Evaluation of LOO, removing best data by bs of 10")
    for i in range(0, 50, 5):
        train_eval=copy.deepcopy(train)
        eliminated=0
        for ss in sorted_results:
            if eliminated == i:
                print(f"Testing with dataset of size of {len(train_eval)}")
                break
            train_eval.data.pop(ss)
            eliminated+=1
        set_seed()
        Classifier = RNN_Classifier(train_eval)
        _, dev_res, _ = Classifier.train_test(train_eval, dev, test)
        print(f"Results of {dev_res}")
    print("Evaluation of LOO, removing worst data by bs of 10")
    for i in range(0, 50, 5):
        train_eval = copy.deepcopy(train)
        eliminated = 0
        for ss in sorted_results[::-1]:
            if eliminated == i:
                print(f"Testing with dataset of size of {len(train_eval)}")
                break
            train_eval.data.pop(ss)
            eliminated += 1
        set_seed()
        Classifier = LogReg_Classifier(train_eval)
        _, dev_res, _ = Classifier.train_test(train_eval, dev, test)
        print(f"Results of {dev_res}")
    # results_train_iter, results_dev_iter, results_test_iter =classifier_algo.train_test(train, dev, test)

#66.74

if __name__ == '__main__':
    main()