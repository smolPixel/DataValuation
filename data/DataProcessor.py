from torch.utils.data import Dataset
import os, io
import numpy as np
import json
import pandas as pd
import torch
from collections import defaultdict, OrderedDict, Counter
from nltk.tokenize import TweetTokenizer, sent_tokenize
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from transformers import BertTokenizer
import math
import ast
import pickle
# import bcolz
import torch
import copy
import random


class OrderedCounter(Counter, OrderedDict):
	"""Counter that remembers the order elements are first encountered"""
	def __repr__(self):
		return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

	def __reduce__(self):
		return self.__class__, (OrderedDict(self),)
def sample_class(df, i, prop, ds):
	"""Sample the class i from the dataframe, with oversampling if needed. """
	size_class=len(df[df['label'] == i])
	num_sample_tot=math.ceil(ds * prop)
	#Sample a first time
	num_sample = min(num_sample_tot, size_class)
	sample = df[df['label'] == i].sample(n=num_sample)
	num_sample_tot-=num_sample
	while num_sample_tot!=0:
		num_sample=min(num_sample_tot, size_class)
		sampleTemp=df[df['label'] == i].sample(n=num_sample)
		sample = pd.concat([sample, sampleTemp])
		num_sample_tot-=num_sample
	return sample



def get_dataFrame(task, dataset_size):
	"""Get the dataframe for the particular split. If it does not exist: create it"""
	create_train=False

	dfVal = pd.read_csv(f'data/{task}/dev.tsv', sep='\t')
	dfTest=pd.read_csv(f'data/{task}/test.tsv', sep='\t')
	dfTrain=pd.read_csv(f'data/{task}/train.tsv', sep='\t').dropna(axis=1)

	num_labels=len(set(dfTrain['label']))

	#Sampling balanced data
	#We always oversample data to eliminate the unbalance factor from the DA algos, as the assumption is that DA is going to be more efficient if the data is unbalanced
	if dataset_size==0:
		max_size = dfTrain['label'].value_counts().max()
		prop=max_size/len(dfTrain)
	else:
		prop=1/num_labels
	NewdfTrain=sample_class(dfTrain, 0, prop, dataset_size)
	for i in range(1, num_labels):
		prop = len(dfTrain[dfTrain['label'] == i]) / len(dfTrain)
		# TODO HERE
		prop = 1 / num_labels
		NewdfTrain=pd.concat([NewdfTrain ,sample_class(dfTrain, i, prop, dataset_size)])
	dfTrain=NewdfTrain
	print(f"Length of the dataframe {len(dfTrain)}")
	return dfTrain, dfVal, dfTest



def initialize_dataset():
	# if argdict['tokenizer'] == "tweetTokenizer":
	# 	tokenizer = TweetTokenizer()
	# elif argdict['tokenizer']=="PtweetTokenizer":
	# 	tokenizer= TweetTokenizer()
	# elif argdict['tokenizer'] == "mwordPiece":
	# 	tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
	# 	num_token_add = tokenizer.add_tokens(['<bos>', '<eos>', '<unk>', '<pad>'], special_tokens=True)  ##This line is updated
	# else:
	# 	raise ValueError("Incorrect tokenizer")

	tokenizer = TweetTokenizer()
	from data.SST2.SST2Dataset import SST2_dataset
	#Textual dataset


	train, dev, test=get_dataFrame('SST2', 500)
	vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence.lower()) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
	vocab.set_default_index(vocab["<unk>"])
	train=SST2_dataset(train, tokenizer, vocab)
	dev=SST2_dataset(dev, tokenizer, vocab)
	test=SST2_dataset(test, tokenizer, vocab)
	# self.input_size=train.vocab_size

	return train, dev, test



def separate_per_class(dataset):
	"""Separate a dataset per class"""
	num_class=len(dataset.argdict['categories'])
	datasets=[copy.deepcopy(dataset) for _ in range(num_class)]
	# print(datasets)
	for ind, ex in dataset.data.items():
		lab=ex['label']
		for i, ds in enumerate(datasets):
			if i!=lab:
				ds.data.pop(ind)
	for ds in datasets:
		ds.reset_index()
	return datasets


