"""
DESCRIPTION

Copyright (C) Weicong Kong, 21/03/2022
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

DATA_ROOT = os.path.join('nbme-score-clinical-patient-notes')

TRAIN_PATH = os.path.join(DATA_ROOT, 'train.csv')
TEST_PATH = os.path.join(DATA_ROOT, 'test.csv')
FEATURE_PATH = os.path.join(DATA_ROOT, 'features.csv')
PATIENT_NOTES_PATH = os.path.join(DATA_ROOT, 'patient_notes.csv')

OUTPUT_DIR = os.path.join('output')
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def get_logger(filename=OUTPUT_DIR + 'train'):
	from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
	logger = getLogger(__name__)
	logger.setLevel(INFO)
	handler1 = StreamHandler()
	handler1.setFormatter(Formatter("%(message)s"))
	handler2 = FileHandler(filename=f"{filename}.log")
	handler2.setFormatter(Formatter("%(message)s"))
	logger.addHandler(handler1)
	logger.addHandler(handler2)
	return logger


LOGGER = get_logger()
