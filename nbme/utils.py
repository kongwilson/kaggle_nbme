"""
DESCRIPTION

Copyright (C) Weicong Kong, 21/03/2022
"""
import ast
import math
import os
import random
import time
import warnings
import re
import numpy as np
import pandas as pd
import tokenizers
import torch
import transformers
from sklearn.metrics import f1_score


warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CFG:

	wandb = False
	competition = 'NBME'
	_wandb_kernel = 'nakama'
	debug = True
	apex = True
	print_freq = 100
	num_workers = 0
	hugging_face_model_name = "microsoft/deberta-base"  # WKNOTE: hugging face model name
	scheduler = 'cosine'  # ['linear', 'cosine']
	batch_scheduler = True
	num_cycles = 0.5
	num_warmup_steps = 0
	epochs = 5
	encoder_lr = 2e-5
	decoder_lr = 2e-5
	min_lr = 1e-6
	eps = 1e-6
	betas = (0.9, 0.999)
	batch_size = 2
	fc_dropout = 0.2
	max_len = 512
	weight_decay = 0.01  # WKNOTE: the regularization parameter (usually the L2 norm of the weights)
	gradient_accumulation_steps = 1
	max_grad_norm = 1000
	seed = 42
	n_fold = 5
	train_fold = [0, 1, 2, 3, 4]
	train = True


if CFG.debug:
	CFG.epochs = 2
	CFG.train_fold = [0]
else:
	CFG.train_fold = list(range(CFG.n_fold))


# WK: need to download the kaggle data and store it in the following `DATA_ROOT` folder in the project

DATA_ROOT = os.path.join('nbme-score-clinical-patient-notes')

TRAIN_PATH = os.path.join(DATA_ROOT, 'train.csv')
TEST_PATH = os.path.join(DATA_ROOT, 'test.csv')
FEATURE_PATH = os.path.join(DATA_ROOT, 'features.csv')
PATIENT_NOTES_PATH = os.path.join(DATA_ROOT, 'patient_notes.csv')

MODEL_STORE = os.path.join('output')
if not os.path.exists(MODEL_STORE):
	os.makedirs(MODEL_STORE, exist_ok=True)


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
		model_name = 'unknow'
	else:
		model_name = replace_suspicious_characters_from_path_name_with_underscore(model_name)
	logger = getLogger()
	logger.setLevel(INFO)
	handler1 = StreamHandler()
	handler1.setFormatter(Formatter("%(message)s"))
	handler2 = FileHandler(filename=os.path.join('log', f"{model_name}.log"))
	handler2.setFormatter(Formatter("%(message)s"))
	logger.addHandler(handler1)
	logger.addHandler(handler2)
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


def replace_suspicious_characters_from_path_name_with_underscore(name):
	return re.sub(r'[^\w\-_\. ]', '_', name).lstrip().rstrip()


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


