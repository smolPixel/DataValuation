"""Calculating values with TMC Shapley, from https://arxiv.org/pdf/1904.02868.pdf"""


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
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def TMC_Shapley(train, dev, test, classifier_algo, dev_baseline):
    #Let's program one iteration. First we need to randomly permute data point
    #dev_baseline= V(D) in paper

    #For now let's put PT at 2%, aka, when we get at 2% of the value of dev_baseline we are satisfied
    PT=0.02
    TRUNC_MAX=5
    ITER_NUM=10

    # train_iter=copy.deepcopy(train)
    # permuatation=train_iter.permute_data()
    #baseline is 0.5 for binary dataset
    vals=[0.5 for i in range(len(train))]
    phis=[0 for i in range(len(train))]
    phis_prec=phis
    t=1
    truncation_counter=0
    for t in range(1, 10, 1):
        train_iter = copy.deepcopy(train)
        permuatation = train_iter.permute_data()
        for j in tqdm(range(1, len(train_iter))):
            if truncation_counter>5 and abs(dev_baseline-vals[j-1])<PT:
                vals[j]=vals[j-1]
                vjt=vals[j]
            else:
                if abs(dev_baseline-vals[j-1])<PT:
                    truncation_counter+=1
                else:
                    truncation_counter=0
                train_trunc=copy.deepcopy(train_iter)
                train_trunc.truncate(permuatation[:j])
                new_point=permuatation[j-1]
                set_seed(seed=t)
                # print(len(train_trunc))
                Classifier = classifier_algo(train_trunc)
                _, vjt, _ = Classifier.train_test(train_trunc, dev, test)
                vals[j]=vjt
                #Inverse to the paper, to keep in track with loo: baseline is with the point included
            phis[new_point]=((t-1)/t)*phis[new_point]+(vals[j-1]-vjt)/t
        phis_prec=phis

    return phis

def main():
    set_seed()
    DATASET_SIZE=100
    train, dev, test=initialize_dataset(DATASET_SIZE)

    print(f"Initialized SST-2 with length of {len(train)}")
    # classifiers=[Bert_Classifier]
    classifiers=[LogReg_Classifier, RNN_Classifier, Bert_Classifier]
    # names=['BERT']
    # names=['LogReg', 'RNN', 'BERT']
    names=['BERT']
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

        shapleys=TMC_Shapley(train, dev, test, classifier_algo, dev_baseline)

        sorted_results=np.argsort(shapleys)
        # print(sorted_results)
        # print(sorted_results[::-1])
        results_remove_best=[dev_baseline]
        results_remove_worst=[dev_baseline]
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

        auc_best=auc(values_x, results_remove_best)
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
        auc_worst=auc(values_x, results_remove_worst)
        print(f"Area under curve is {auc_worst}")

        print(f"Final metric: {auc_worst-auc_best}")

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
        plt.title(f'Graphes/{name}-TMC')
        plt.savefig(f'Graphes/{name}TMC.png')


if __name__ == '__main__':
    main()


# Original code for reference:

    # def _tmc_shap(self, iterations, tolerance=None, sources=None):
    #     """Runs TMC-Shapley algorithm.
    #
    #     Args:
    #         iterations: Number of iterations to run.
    #         tolerance: Truncation tolerance ratio.
    #         sources: If values are for sources of data points rather than
    #                individual points. In the format of an assignment array
    #                or dict.
    #     """
    #     if sources is None:
    #         sources = {i: np.array([i]) for i in range(len(self.X))}
    #     elif not isinstance(sources, dict):
    #         sources = {i: np.where(sources == i)[0] for i in set(sources)}
    #     model = self.model
    #     try:
    #         self.mean_score
    #     except:
    #         self._tol_mean_score()
    #     if tolerance is None:
    #         tolerance = self.tolerance
    #     marginals, idxs = [], []
    """Main loop of iterations"""
    #     for iteration in range(iterations):
    #         if 10 * (iteration + 1) / iterations % 1 == 0:
    #             print('{} out of {} TMC_Shapley iterations.'.format(
    #                 iteration + 1, iterations))
    #         marginals, idxs = self.one_iteration(
    #             tolerance=tolerance,
    #             sources=sources
    #         )
    """It seems like instead of doing a running average here he keeps all iters and do the average at the end"""
    #         self.mem_tmc = np.concatenate([
    #             self.mem_tmc,
    #             np.reshape(marginals, (1, -1))
    #         ])
    #         self.idxs_tmc = np.concatenate([
    #             self.idxs_tmc,
    #             np.reshape(idxs, (1, -1))
    #         ])

    # def one_iteration(self, tolerance, sources=None):
    #     """Runs one iteration of TMC-Shapley algorithm."""
    #     if sources is None:
    #         sources = {i: np.array([i]) for i in range(len(self.X))}
    #     elif not isinstance(sources, dict):
    #         sources = {i: np.where(sources == i)[0] for i in set(sources)}
    """Permutation step"""
    #     idxs = np.random.permutation(len(sources))
    """Initializing shapley values at 0"""
    #     marginal_contribs = np.zeros(len(self.X))
    #     X_batch = np.zeros((0,) + tuple(self.X.shape[1:]))
    #     y_batch = np.zeros(0, int)
    """Not sure what sample_weight_batch is, not mentionned in paper"""
    #     sample_weight_batch = np.zeros(0)
    #     truncation_counter = 0
    #     new_score = self.random_score
    #     for n, idx in enumerate(idxs):
    """TODO UNDERSTAND WHERE HE DOES RUNNING AVERAGE"""
    #         old_score = new_score
    """Add X Y to the batch"""
    #         X_batch = np.concatenate([X_batch, self.X[sources[idx]]])
    #         y_batch = np.concatenate([y_batch, self.y[sources[idx]]])
    #         if self.sample_weight is None:
    #             sample_weight_batch = None
    #         else:
    #             sample_weight_batch = np.concatenate([
    #                 sample_weight_batch,
    #                 self.sample_weight[sources[idx]]
    #             ])
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")
    #             if (self.is_regression
    #                 or len(set(y_batch)) == len(set(self.y_test))): ##FIXIT
    #                 self.restart_model()
    #                 if sample_weight_batch is None:
    #                     self.model.fit(X_batch, y_batch)
    #                 else:
    #                     self.model.fit(
    #                         X_batch,
    #                         y_batch,
    #                         sample_weight = sample_weight_batch
    #                     )
    """Calculate new score"""
    #                 new_score = self.value(self.model, metric=self.metric)
    """This is the second part of running average """
    #         marginal_contribs[sources[idx]] = (new_score - old_score)
    #         marginal_contribs[sources[idx]] /= len(sources[idx])
    #         distance_to_full_score = np.abs(new_score - self.mean_score)
    """Truncate if fulfill truncation condition 5 times in a row TODO"""
    #         if distance_to_full_score <= tolerance * self.mean_score:
    #             truncation_counter += 1
    #             if truncation_counter > 5:
    #                 break
    #         else:
    #             truncation_counter = 0
    #     return marginal_contribs, idxs