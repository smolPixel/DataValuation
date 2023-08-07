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

    print("Running LOO with LogReg classifier")

    from Classifier.LogReg import LogReg_Classifier
    set_seed()
    Classifier = LogReg_Classifier(train)
    results=Classifier.train_test(train, dev, test)
    dev_baseline=results[1]

    print(f"Results with all data is {results}")

    results=[]
    for i in range(DATASET_SIZE):
        train_loo=copy.deepcopy(train)
        train_loo.data.pop(i)
        set_seed()
        Classifier = LogReg_Classifier(train_loo)
        _, dev_res, _ = Classifier.train_test(train_loo, dev, test)
        results.append(dev_res-dev_baseline)

    print(results)

    # results_train_iter, results_dev_iter, results_test_iter =classifier_algo.train_test(train, dev, test)

#66.74

if __name__ == '__main__':
    main()