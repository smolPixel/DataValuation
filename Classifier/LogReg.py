import os
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import itertools

class LogReg_Classifier():
    """Logistic regression with TF IDF"""

    def __init__(self, train):
        self.train=train
        self.init_model()

    def init_model(self):
        sents=list(self.train.return_pandas()['sentence'])
        self.vectorizer=TfidfVectorizer()
        self.vectorizer.fit(sents)
        self.clf=LogisticRegression()

    def train_test(self, train, dev, test):
        train=train.return_pandas()
        X=self.vectorizer.transform(list(train['sentence']))
        Y=list(train['label'])
        self.clf.fit(X, Y)
        preds=self.clf.predict(X)
        acc_train=accuracy_score(Y, preds)
        """dev"""
        dev=dev.return_pandas()
        X = self.vectorizer.transform(list(dev['sentence']))
        Y=list(dev['label'])
        preds = self.clf.predict(X)
        acc_dev = accuracy_score(Y, preds)
        """test"""
        test=test.return_pandas()
        X = self.vectorizer.transform(list(test['sentence']))
        Y=list(test['label'])
        preds = self.clf.predict(X)
        acc_test = accuracy_score(Y, preds)
        return acc_train, acc_dev, acc_test