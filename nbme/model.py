"""
DESCRIPTION

Copyright (C) Weicong Kong, 30/03/2022
"""
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.deberta.tokenization_deberta_fast import DebertaTokenizerFast
import wandb

from nbme.utils import *
from nbme.data_loader import ProjectDataLoader


# ====================================================
# Model(s)
# ====================================================
class HuggingFaceBackedModel(nn.Module):

	def __init__(self, hugging_face_model_name, fc_dropout, config_path=None, pretrained=False, cache_dir=None):

		super().__init__()
		if config_path is None:
			self.config = AutoConfig.from_pretrained(
				hugging_face_model_name, output_hidden_states=True, cache_dir=cache_dir)
		else:
			self.config = torch.load(config_path)
		if pretrained:
			self.model = AutoModel.from_pretrained(
				hugging_face_model_name, config=self.config, cache_dir=cache_dir)
		else:
			self.model = AutoModel(self.config)
		self.fc_dropout = nn.Dropout(fc_dropout)
		self.fc = nn.Linear(self.config.hidden_size, 1)  # WK: map each token vector to a prob of True/False
		self._init_weights(self.fc)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
			if module.padding_idx is not None:
				module.weight.data[module.padding_idx].zero_()
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)

	def feature(self, inputs):
		# WKNOTE: https://huggingface.co/transformers/v4.7.0/model_doc/deberta.html
		#   The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top
		outputs = self.model(**inputs)
		last_hidden_states = outputs[0]
		return last_hidden_states

	def forward(self, inputs):
		feature = self.feature(inputs)
		output = self.fc(self.fc_dropout(feature))
		return output


class Trainer(object):

	def __init__(
			self, train_config, project_data_loader: ProjectDataLoader,
			model_store_dir=None, pretrained_cache_dir=None):

		if model_store_dir is None:
			model_store_dir = 'model_store'

		if pretrained_cache_dir is None:
			pretrained_cache_dir = 'cache'

		if not os.path.exists(model_store_dir):
			os.makedirs(model_store_dir, exist_ok=True)

		if not os.path.exists(pretrained_cache_dir):
			os.makedirs(pretrained_cache_dir, exist_ok=True)

		model_folder = os.path.join(
			model_store_dir, replace_suspicious_characters_from_path_name_with_underscore(
				train_config.hugging_face_model_name))
		if not os.path.exists(model_folder):
			os.makedirs(model_folder, exist_ok=True)

		self.config = train_config
		self.model_store_dir = model_store_dir
		self.pretrained_cache_dir = pretrained_cache_dir
		self.model_folder = model_folder
		self.tokenizer = self.load_tokenizer()
		self.project_data_loader = project_data_loader

		# ====================================================
		# wandb
		# ====================================================
		if self.config.wandb:

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
				name=self.config.hugging_face_model_name,
				config=class2dict(self.config),
				group=self.config.hugging_face_model_name,
				job_type="train",
				anonymous=anony)
		return

	def load_tokenizer(self):
		tokenizer = AutoTokenizer.from_pretrained(self.config.hugging_face_model_name, cache_dir=self.pretrained_cache_dir)
		tokenizer.save_pretrained(
			os.path.join(self.model_folder, 'tokenizer')
		)
		return tokenizer

	def analyse_the_max_len_from_the_training_data(self, train_data):
		"""
		update the self.config.max_len, or the train_data to align with the max_len param given in the config
		:param train_data:
		:return:
		"""
		logger = get_logger(self.config.hugging_face_model_name)

		patient_notes = train_data['pn_history'].fillna('').unique()
		features = train_data['feature_text'].fillna('').unique()

		pn_history_lengths = []
		# WKNOTE: the usage of the tqdm wrapper from tqdm.auto, which wrap a collection with tqdm progress tracking
		pn_collections = tqdm(patient_notes, total=len(patient_notes))
		for text in pn_collections:
			length = len(self.tokenizer(text, add_special_tokens=False)['input_ids'])
			pn_history_lengths.append(length)
		logger.info(f'pn_history max(lengths): {max(pn_history_lengths)}')

		features_lengths = []
		feature_collections = tqdm(features, total=len(features))
		for text in feature_collections:
			length = len(self.tokenizer(text, add_special_tokens=False)['input_ids'])
			features_lengths.append(length)
		logger.info(f'feature_text max(lengths): {max(features_lengths)}')

		# WKNOTE: update the max length setting for the training data using the corresponding backbone hugging face model
		train_data_max_len = max(pn_history_lengths) + max(features_lengths) + 3  # cls & sep & sep
		if train_data_max_len < self.config.max_len:
			self.config.max_len = train_data_max_len

		logger.info(f"max_len: {self.config.max_len}")

	def train_loop(self, folds, fold, model_constructor):
		logger = get_logger(self.config.hugging_face_model_name)
		logger.info(f"========== fold: {fold} training ==========")

		# ====================================================
		# loader
		# ====================================================
		train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
		valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
		valid_texts = valid_folds['pn_history'].values
		valid_labels = create_labels_for_scoring(valid_folds)

		train_dataset = TrainDataset(self.tokenizer, train_folds, self.config.max_len)
		valid_dataset = TrainDataset(self.tokenizer, valid_folds, self.config.max_len)

		train_loader = DataLoader(
			train_dataset,
			batch_size=self.config.batch_size,
			shuffle=True,
			num_workers=self.config.num_workers, pin_memory=True, drop_last=True)
		valid_loader = DataLoader(
			valid_dataset,
			batch_size=self.config.batch_size,
			shuffle=False,
			num_workers=self.config.num_workers, pin_memory=True, drop_last=False)

		# ====================================================
		# model & optimizer
		# ====================================================
		model = model_constructor(
			self.config.hugging_face_model_name, self.config.fc_dropout, config_path=None, pretrained=True)
		# WKNOTE:save model configuration as `config.pth`
		torch.save(model.config, os.path.join(self.model_folder, 'config.pth'))
		# model = nn.DataParallel(model)
		model.to(DEVICE)

		def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
			# TODO: the purpose of this function should be reviewed. It may apply to deberta model only
			# param_optimizer = list(model.named_parameters())
			no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
			if isinstance(model, nn.DataParallel):
				optimizer_parameters = [
					{'params': [p for n, p in model.module.model.named_parameters() if
						not any(nd in n for nd in no_decay)],
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
			encoder_lr=self.config.encoder_lr,
			decoder_lr=self.config.decoder_lr,
			weight_decay=self.config.weight_decay)
		optimizer = AdamW(optimizer_parameters, lr=self.config.encoder_lr, eps=self.config.eps, betas=self.config.betas)

		num_train_steps = int(len(train_folds) / self.config.batch_size * self.config.epochs)
		scheduler = self.get_scheduler(optimizer, num_train_steps)

		# ====================================================
		# loop
		# ====================================================
		criterion = nn.BCEWithLogitsLoss(reduction="none")

		best_score = 0.
		# WKNOTE: for staging the best model `weights` and `predictions`
		model_checkpoint_path = os.path.join(self.model_folder, f'fold_{fold}.pth')

		for epoch in range(self.config.epochs):

			start_time = time.time()

			# train
			avg_loss = self.train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, DEVICE)

			# eval
			avg_val_loss, predictions = self.valid_fn(valid_loader, model, criterion, DEVICE)
			predictions = predictions.reshape((len(valid_folds), self.config.max_len))

			# scoring
			char_probs = get_char_probs(valid_texts, predictions, self.tokenizer)
			results = get_results(char_probs, th=0.5)
			preds = get_predictions(results)
			score = get_score(valid_labels, preds)

			elapsed = time.time() - start_time

			logger.info(
				f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
			logger.info(f'Epoch {epoch + 1} - Score: {score:.4f}')
			if self.config.wandb:
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
		valid_folds[[i for i in range(self.config.max_len)]] = predictions

		torch.cuda.empty_cache()
		gc.collect()

		return valid_folds, best_score

	def train_fn(self, fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
		model.train()
		scaler = torch.cuda.amp.GradScaler(enabled=self.config.apex)
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
			with torch.cuda.amp.autocast(enabled=self.config.apex):
				y_preds = model(inputs)
			loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
			loss = torch.masked_select(
				loss, labels.view(-1, 1) != -1).mean()  # WKNOTE: only eval the loss for non-padding
			if self.config.gradient_accumulation_steps > 1:
				loss = loss / self.config.gradient_accumulation_steps
			losses.update(loss.item(), batch_size)
			scaler.scale(loss).backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
			if (step + 1) % self.config.gradient_accumulation_steps == 0:
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()
				global_step += 1
				if self.config.batch_scheduler:
					scheduler.step()
			if step % self.config.print_freq == 0 or step == (len(train_loader) - 1):
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
			if self.config.wandb:
				wandb.log({
					f"[fold{fold}] loss": losses.val,
					f"[fold{fold}] lr": scheduler.get_lr()[0]})
		return losses.avg

	def valid_fn(self, valid_loader, model, criterion, device):
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
			if self.config.gradient_accumulation_steps > 1:
				loss = loss / self.config.gradient_accumulation_steps
			losses.update(loss.item(), batch_size)
			preds.append(y_preds.sigmoid().to('cpu').numpy())
			end = time.time()
			if step % self.config.print_freq == 0 or step == (len(valid_loader) - 1):
				print(
					'EVAL: [{0}/{1}] '
					'Elapsed {remain:s} '
					'Loss: {loss.val:.4f}({loss.avg:.4f}) '.format(
						step, len(valid_loader),
						loss=losses,
						remain=time_since(start, float(step + 1) / len(valid_loader))))
		predictions = np.concatenate(preds)
		return losses.avg, predictions

	def inference_fn(self, test_loader, model, device):
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

	def get_scheduler(self, optimizer, num_train_steps):
		if self.config.scheduler == 'linear':
			scheduler = get_linear_schedule_with_warmup(
				optimizer, num_warmup_steps=self.config.num_warmup_steps, num_training_steps=num_train_steps
			)
		elif self.config.scheduler == 'cosine':
			scheduler = get_cosine_schedule_with_warmup(
				optimizer, num_warmup_steps=self.config.num_warmup_steps, num_training_steps=num_train_steps,
				num_cycles=self.config.num_cycles
			)
		else:
			raise NotImplementedError()
		return scheduler

	def get_result(self, oof_df):
		logger = get_logger(self.config.hugging_face_model_name)
		labels = create_labels_for_scoring(oof_df)
		predictions = oof_df[[i for i in range(self.config.max_len)]].values
		char_probs = get_char_probs(oof_df['pn_history'].values, predictions, self.tokenizer)
		results = get_results(char_probs, th=0.5)
		preds = get_predictions(results)
		score = get_score(labels, preds)
		logger.info(f'Score: {score:<.4f}')
