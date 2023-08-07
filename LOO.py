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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed()
    train, dev, test=initialize_dataset()
    print(f"Initialized SST-2 with length of {len(train)}")

    print("Running LOO with LogReg classifier")

    from Classifier.LogReg import LogReg_Classifier
    Classifier = LogReg_Classifier(train)
    results=Classifier.train_test(train, dev, test)

    print(f"Results with all data is {results}")

    # results_train_iter, results_dev_iter, results_test_iter =classifier_algo.train_test(train, dev, test)

#66.74

if __name__ == '__main__':
    main()