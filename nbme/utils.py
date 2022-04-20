"""
DESCRIPTION

Copyright (C) Weicong Kong, 21/03/2022
"""
import ast
import math
import os
import gc
import random
import time
import warnings
import re
import numpy as np
import pandas as pd
import tokenizers
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import itertools

from dotenv import load_dotenv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import transformers
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from nbme.data_loader import TrainDataset

warnings.filterwarnings("ignore")

load_dotenv()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def replace_suspicious_characters_from_path_name_with_underscore(name):
	return re.sub(r'[^\w\-_\. ]', '_', name).lstrip().rstrip()


def class2dict(f):
	return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


def micro_f1(preds, truths):
	"""
	Micro f1 on binary arrays.
	:param preds:   (list of ndarray of ints): Predictions.
	:param truths:  (list of ndarray of ints): Ground truths.
	:return:        float: f1 score.
	"""
	# Micro : aggregating over all instances
	preds = np.concatenate(preds)
	truths = np.concatenate(truths)
	return f1_score(truths, preds)


def spans_to_binary(spans, length=None):
	"""
	Converts spans to a binary array indicating whether each character is in the span.

	:param spans:
	:param length:
	:return:        np array [length] Binarized spans
	"""
	length = np.max(spans) if length is None else length
	binary = np.zeros(length)
	for start, end in spans:
		binary[start:end] = 1
	return binary


def span_micro_f1(preds, truths):
	"""
	Micro f1 on spans.
	:param preds:   (list of lists of two ints): Prediction spans
	:param truths:  (list of lists of two ints): Ground truth spans.
	:return:        float: f1 score.
	"""
	bin_preds = []
	bin_truths = []
	for pred, truth in zip(preds, truths):
		if not len(pred) and not len(truth):
			continue
		length = max(np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0)
		bin_preds.append(spans_to_binary(pred, length))
		bin_truths.append(spans_to_binary(truth, length))
	return micro_f1(bin_preds, bin_truths)


def get_score(y_true, y_pred):
	score = span_micro_f1(y_true, y_pred)
	return score


def get_logger(model_name=None):
	from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
	if not os.path.exists('log'):
		os.makedirs('log', exist_ok=True)
	if not model_name:
		model_name = 'unknown'
	else:
		model_name = replace_suspicious_characters_from_path_name_with_underscore(model_name)
	logger = getLogger()
	logger.setLevel(INFO)
	formatter = Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(lineno)d: %(message)s')
	# formatter = Formatter("%(message)s")  # WK: lite
	console_handler = StreamHandler()
	console_handler.setFormatter(formatter)
	console_handler.name = 'console'
	file_handler = FileHandler(filename=os.path.join('log', f"{model_name}.log"))
	file_handler.setFormatter(formatter)
	file_handler.name = 'file'
	handler_names = [h.name for h in logger.handlers]
	if console_handler.name not in handler_names:
		logger.addHandler(console_handler)
	if file_handler.name not in handler_names:
		logger.addHandler(file_handler)
	return logger


def seed_everything(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True


# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def as_minutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def time_since(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (remain %s)' % (as_minutes(s), as_minutes(rs))


def create_labels_for_scoring(df):
	# example: ['0 1', '3 4'] -> ['0 1; 3 4']
	df['location_for_create_labels'] = [ast.literal_eval(f'[]')] * len(df)
	for i in range(len(df)):
		lst = df.loc[i, 'location']
		if lst:
			new_lst = ';'.join(lst)
			df.loc[i, 'location_for_create_labels'] = ast.literal_eval(f'[["{new_lst}"]]')
	# create labels
	truths = []
	for location_list in df['location_for_create_labels'].values:
		truth = []
		if len(location_list) > 0:
			location = location_list[0]
			for loc in [s.split() for s in location.split(';')]:
				start, end = int(loc[0]), int(loc[1])
				truth.append([start, end])
		truths.append(truth)
	return truths


def get_char_probs(texts, predictions, tokenizer):
	# WKNOTE: from tokens feature matching to character feature matching
	results = [np.zeros(len(t)) for t in texts]
	for i, (text, prediction) in enumerate(zip(texts, predictions)):
		encoded = tokenizer(
			text, add_special_tokens=True, return_offsets_mapping=True)
		for idx, (offset_mapping, pred) in enumerate(zip(encoded['offset_mapping'], prediction)):
			start = offset_mapping[0]
			end = offset_mapping[1]
			results[i][start:end] = pred
	return results


def get_results(char_probs, th=0.5):
	results = []
	for char_prob in char_probs:
		result = np.where(char_prob >= th)[0] + 1
		result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
		result = [f"{min(r)} {max(r)}" for r in result]
		result = ";".join(result)
		results.append(result)
	return results


def get_predictions(results):
	predictions = []
	for result in results:
		prediction = []
		if result != "":
			for loc in [s.split() for s in result.split(';')]:
				start, end = int(loc[0]), int(loc[1])
				prediction.append([start, end])
		predictions.append(prediction)
	return predictions



# ====================================================
# train loop
# ====================================================


