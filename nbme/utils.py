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


# WK: the global configuration var
class CFG:
	wandb = False
	competition = 'NBME'
	_wandb_kernel = 'nakama'
	debug = True
	apex = True
	print_freq = 100
	num_workers = 0
	hugging_face_model_name = "microsoft/deberta-large"  # WKNOTE: hugging face model name
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
	train_folds = [0, 1, 2, 3, 4]
	train = True
	tokenizer = None  # wait to be loaded


if CFG.debug:
	CFG.epochs = 2
	CFG.train_folds = [0]
else:
	CFG.train_folds = list(range(CFG.n_fold))

# WK: need to download the kaggle data and store it in the following `DATA_ROOT` folder in the project

DATA_ROOT = os.path.join('nbme-score-clinical-patient-notes')

TRAIN_PATH = os.path.join(DATA_ROOT, 'train.csv')
TEST_PATH = os.path.join(DATA_ROOT, 'test.csv')
FEATURE_PATH = os.path.join(DATA_ROOT, 'features.csv')
PATIENT_NOTES_PATH = os.path.join(DATA_ROOT, 'patient_notes.csv')

MODEL_STORE = os.path.join('model_store')
PRETRAINED_CACHE = os.path.join('cache')
if not os.path.exists(MODEL_STORE):
	os.makedirs(MODEL_STORE, exist_ok=True)

if not os.path.exists(PRETRAINED_CACHE):
	os.makedirs(PRETRAINED_CACHE, exist_ok=True)


def replace_suspicious_characters_from_path_name_with_underscore(name):
	return re.sub(r'[^\w\-_\. ]', '_', name).lstrip().rstrip()


MODEL_FOLDER = os.path.join(
	MODEL_STORE, replace_suspicious_characters_from_path_name_with_underscore(CFG.hugging_face_model_name))
if not os.path.exists(MODEL_FOLDER):
	os.makedirs(MODEL_FOLDER, exist_ok=True)


def class2dict(f):
	return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

# ====================================================
# wandb
# ====================================================
if CFG.wandb:

	import wandb

	try:

		wandb_api_key = os.getenv('WANDB_API_KEY')
		wandb.login(key=wandb_api_key)
		anony = None
	except:
		anony = "must"
		print(
			'wandb connection failed')

	run = wandb.init(
		project='NBME-Public',
		name=CFG.hugging_face_model_name,
		config=class2dict(CFG),
		group=CFG.hugging_face_model_name,
		job_type="train",
		anonymous=anony)


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


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
	model.train()
	scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
	losses = AverageMeter()
	start = time.time()
	global_step = 0
	for step, (inputs, labels) in enumerate(train_loader):
		for k, v in inputs.items():
			inputs[k] = v.to(device)
		labels = labels.to(device)
		batch_size = labels.size(0)
		# WKNOTE: https://pytorch.org/docs/stable/amp.html#torch.autocast
		#   Instances of autocast serve as context managers or decorators that allow regions
		#   of your script to run in mixed precision.
		#   In these regions, ops run in an op-specific dtype chosen by autocast to improve performance while
		#   maintaining accuracy.
		with torch.cuda.amp.autocast(enabled=CFG.apex):
			y_preds = model(inputs)
		loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
		loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()  # WKNOTE: only eval the loss for non-padding
		if CFG.gradient_accumulation_steps > 1:
			loss = loss / CFG.gradient_accumulation_steps
		losses.update(loss.item(), batch_size)
		scaler.scale(loss).backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
		if (step + 1) % CFG.gradient_accumulation_steps == 0:
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad()
			global_step += 1
			if CFG.batch_scheduler:
				scheduler.step()
		if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
			print(
				'Epoch: [{0}][{1}/{2}] '
				'Elapsed {remain:s} '
				'Loss: {loss.val:.4f}({loss.avg:.4f}) '
				'Grad: {grad_norm:.4f}  '
				'LR: {lr:.8f}  '.format(
					epoch + 1, step, len(train_loader),
					remain=time_since(start, float(step + 1) / len(train_loader)),
					loss=losses,
					grad_norm=grad_norm,
					lr=scheduler.get_lr()[0]))
		if CFG.wandb:
			wandb.log({
				f"[fold{fold}] loss": losses.val,
				f"[fold{fold}] lr": scheduler.get_lr()[0]})
	return losses.avg


def valid_fn(valid_loader, model, criterion, device):
	losses = AverageMeter()
	model.eval()
	preds = []
	start = end = time.time()
	for step, (inputs, labels) in enumerate(valid_loader):
		for k, v in inputs.items():
			inputs[k] = v.to(device)
		labels = labels.to(device)
		batch_size = labels.size(0)
		with torch.no_grad():
			y_preds = model(inputs)
		loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
		loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
		if CFG.gradient_accumulation_steps > 1:
			loss = loss / CFG.gradient_accumulation_steps
		losses.update(loss.item(), batch_size)
		preds.append(y_preds.sigmoid().to('cpu').numpy())
		end = time.time()
		if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
			print(
				'EVAL: [{0}/{1}] '
				'Elapsed {remain:s} '
				'Loss: {loss.val:.4f}({loss.avg:.4f}) '.format(
					step, len(valid_loader),
					loss=losses,
					remain=time_since(start, float(step + 1) / len(valid_loader))))
	predictions = np.concatenate(preds)
	return losses.avg, predictions


def inference_fn(test_loader, model, device):
	preds = []
	model.eval()
	model.to(device)
	tk0 = tqdm(test_loader, total=len(test_loader))
	for inputs in tk0:
		for k, v in inputs.items():
			inputs[k] = v.to(device)
		with torch.no_grad():
			y_preds = model(inputs)
		preds.append(y_preds.sigmoid().to('cpu').numpy())
	predictions = np.concatenate(preds)
	return predictions


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
# scheduler
# ====================================================
def get_scheduler(cfg, optimizer, num_train_steps):
	if cfg.scheduler == 'linear':
		scheduler = get_linear_schedule_with_warmup(
			optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
		)
	elif cfg.scheduler == 'cosine':
		scheduler = get_cosine_schedule_with_warmup(
			optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
			num_cycles=cfg.num_cycles
		)
	else:
		raise NotImplementedError()
	return scheduler


# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold, model_constructor):
	logger = get_logger(CFG.hugging_face_model_name)
	logger.info(f"========== fold: {fold} training ==========")

	# ====================================================
	# loader
	# ====================================================
	train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
	valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
	valid_texts = valid_folds['pn_history'].values
	valid_labels = create_labels_for_scoring(valid_folds)

	train_dataset = TrainDataset(CFG.tokenizer, train_folds, CFG.max_len)
	valid_dataset = TrainDataset(CFG.tokenizer, valid_folds, CFG.max_len)

	train_loader = DataLoader(
		train_dataset,
		batch_size=CFG.batch_size,
		shuffle=True,
		num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
	valid_loader = DataLoader(
		valid_dataset,
		batch_size=CFG.batch_size,
		shuffle=False,
		num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

	# ====================================================
	# model & optimizer
	# ====================================================
	model = model_constructor(CFG, config_path=None, pretrained=True)
	# WKNOTE:save model configuration as `config.pth`
	torch.save(model.config, os.path.join(MODEL_FOLDER, 'config.pth'))
	# model = nn.DataParallel(model)
	model.to(DEVICE)

	def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
		# TODO: the purpose of this function should be reviewed. It may apply to deberta model only
		# param_optimizer = list(model.named_parameters())
		no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
		if isinstance(model, nn.DataParallel):
			optimizer_parameters = [
				{'params': [p for n, p in model.module.model.named_parameters() if not any(nd in n for nd in no_decay)],
					'lr': encoder_lr, 'weight_decay': weight_decay},
				{'params': [p for n, p in model.module.model.named_parameters() if any(nd in n for nd in no_decay)],
					'lr': encoder_lr, 'weight_decay': 0.0},
				{'params': [p for n, p in model.module.named_parameters() if "model" not in n],
					'lr': decoder_lr, 'weight_decay': 0.0}
			]
		else:
			optimizer_parameters = [
				{'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
					'lr': encoder_lr, 'weight_decay': weight_decay},
				{'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
					'lr': encoder_lr, 'weight_decay': 0.0},
				{'params': [p for n, p in model.named_parameters() if "model" not in n],
					'lr': decoder_lr, 'weight_decay': 0.0}
			]
		return optimizer_parameters

	optimizer_parameters = get_optimizer_params(
		model,
		encoder_lr=CFG.encoder_lr,
		decoder_lr=CFG.decoder_lr,
		weight_decay=CFG.weight_decay)
	optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

	num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
	scheduler = get_scheduler(CFG, optimizer, num_train_steps)

	# ====================================================
	# loop
	# ====================================================
	criterion = nn.BCEWithLogitsLoss(reduction="none")

	best_score = 0.
	# WKNOTE: for staging the best model `weights` and `predictions`
	model_checkpoint_path = os.path.join(MODEL_FOLDER, f'fold_{fold}.pth')

	for epoch in range(CFG.epochs):

		start_time = time.time()

		# train
		avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, DEVICE)

		# eval
		avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, DEVICE)
		predictions = predictions.reshape((len(valid_folds), CFG.max_len))

		# scoring
		char_probs = get_char_probs(valid_texts, predictions, CFG.tokenizer)
		results = get_results(char_probs, th=0.5)
		preds = get_predictions(results)
		score = get_score(valid_labels, preds)

		elapsed = time.time() - start_time

		logger.info(
			f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
		logger.info(f'Epoch {epoch + 1} - Score: {score:.4f}')
		if CFG.wandb:
			wandb.log({
				f"[fold{fold}] epoch": epoch + 1,
				f"[fold{fold}] avg_train_loss": avg_loss,
				f"[fold{fold}] avg_val_loss": avg_val_loss,
				f"[fold{fold}] score": score})

		if best_score < score:
			best_score = score
			logger.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
			torch.save({
				'model': model.state_dict(),
				'predictions': predictions
			}, model_checkpoint_path)

	predictions = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))['predictions']
	valid_folds[[i for i in range(CFG.max_len)]] = predictions

	torch.cuda.empty_cache()
	gc.collect()

	return valid_folds, best_score


def get_result(oof_df):
	logger = get_logger(CFG.hugging_face_model_name)
	labels = create_labels_for_scoring(oof_df)
	predictions = oof_df[[i for i in range(CFG.max_len)]].values
	char_probs = get_char_probs(oof_df['pn_history'].values, predictions, CFG.tokenizer)
	results = get_results(char_probs, th=0.5)
	preds = get_predictions(results)
	score = get_score(labels, preds)
	logger.info(f'Score: {score:<.4f}')
