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
def sample_class(df, i, prop, argdict):
	"""Sample the class i from the dataframe, with oversampling if needed. """
	size_class=len(df[df['label'] == i])
	ds=argdict['dataset_size'] if argdict['dataset_size']!=0 else len(df)
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



def get_dataFrame(argdict):
	"""Get the dataframe for the particular split. If it does not exist: create it"""
	task=argdict['dataset']
	create_train=False

	#IF we fix the training set, we always want the same
	if argdict['fix_dataset'] and os.path.isfile(f'Selecteddata/{task}/{argdict["dataset_size"]}/train.tsv'):
		# pathTrain = f"SelectedData/{argdict['dataset']}/{argdict['dataset_size']}"
		dfVal = pd.read_csv(f'data/{task}/dev.tsv', sep='\t')
		dfTest=pd.read_csv(f'data/{task}/test.tsv', sep='\t')
		dfTrain=pd.read_csv(f'Selecteddata/{task}/{argdict["dataset_size"]}/train.tsv', sep='\t').dropna(axis=1)
		# print(dfTrain)
		return dfTrain, dfVal, dfTest
	else:
		# pathTrain = f"SelectedData/{argdict['dataset']}/{argdict['dataset_size']}"
		dfVal = pd.read_csv(f'data/{task}/dev.tsv', sep='\t')
		dfTest=pd.read_csv(f'data/{task}/test.tsv', sep='\t')
		dfTrain=pd.read_csv(f'data/{task}/train.tsv', sep='\t').dropna(axis=1)

	# pd.set_option('display.max_rows', None)
	# pd.set_option('display.max_columns', None)
	# pd.set_option('display.width', None)
	# pd.set_option('display.max_colwidth', -1)
	#Sampling balanced data
	#We always oversample data to eliminate the unbalance factor from the DA algos, as the assumption is that DA is going to be more efficient if the data is unbalanced
	print(list(dfTrain))
	if argdict['dataset_size']==0:
		max_size = dfTrain['label'].value_counts().max()
		prop=max_size/len(dfTrain)
	else:
		prop=1/len(argdict['categories'])
	NewdfTrain=sample_class(dfTrain, 0, prop, argdict)
	for i in range(1, len(argdict['categories'])):
		prop = len(dfTrain[dfTrain['label'] == i]) / len(dfTrain)
		# TODO HERE
		prop = 1 / len(argdict['categories'])
		NewdfTrain=pd.concat([NewdfTrain ,sample_class(dfTrain, i, prop, argdict)])
	dfTrain=NewdfTrain
	# Path(pathTrain).mkdir(parents=True, exist_ok=True)
	# dfTrain.to_csv(f"{pathTrain}/train.tsv", sep='\t')
	# fdasklj
	print(f"Length of the dataframe {len(dfTrain)}")
	if argdict['fix_dataset']:
		dfTrain.to_csv(f'Selecteddata/{task}/{argdict["dataset_size"]}/train.tsv', sep='\t')

	return dfTrain, dfVal, dfTest



def initialize_dataset(argdict):
	if argdict['tokenizer'] == "tweetTokenizer":
		tokenizer = TweetTokenizer()
	elif argdict['tokenizer']=="PtweetTokenizer":
		tokenizer= TweetTokenizer()
	elif argdict['tokenizer'] == "mwordPiece":
		tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
		num_token_add = tokenizer.add_tokens(['<bos>', '<eos>', '<unk>', '<pad>'], special_tokens=True)  ##This line is updated
	else:
		raise ValueError("Incorrect tokenizer")


	from data.SST2.SST2Dataset import SST2_dataset
	#Textual dataset


	train, dev, test=get_dataFrame(argdict)
	vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence.lower()) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
	vocab.set_default_index(vocab["<unk>"])
	train=SST2_dataset(train, tokenizer, vocab, argdict)
	dev=SST2_dataset(dev, tokenizer, vocab, argdict)
	test=SST2_dataset(test, tokenizer, vocab, argdict)
	argdict['input_size']=train.vocab_size

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


