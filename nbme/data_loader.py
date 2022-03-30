"""
DESCRIPTION

Copyright (C) Weicong Kong, 30/03/2022
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def prepare_input(tokenizer, text, feature_text, max_len):
	inputs = tokenizer(
		text, feature_text,
		add_special_tokens=True,
		max_length=max_len,
		padding="max_length",
		return_offsets_mapping=False)
	for k, v in inputs.items():
		inputs[k] = torch.tensor(v, dtype=torch.long)
	return inputs


# WKNOTE: a label is a 1-d tensor of `max_len`, with the label texts being 1, others being 0, and [pad] being -1
def create_label(tokenizer, text, annotation_length, location_list, max_len):
	encoded = tokenizer(
		text,
		add_special_tokens=True,
		max_length=max_len,
		padding="max_length",
		return_offsets_mapping=True)
	offset_mapping = encoded['offset_mapping']
	ignore_idxes = np.where(np.array(encoded.sequence_ids()) != 0)[0]
	label = np.zeros(len(offset_mapping))
	label[ignore_idxes] = -1
	if annotation_length != 0:
		for location in location_list:
			for loc in [s.split() for s in location.split(';')]:  # WKNOTE: handling the ';' in the location labels
				start_idx = -1
				end_idx = -1
				start, end = int(loc[0]), int(loc[1])
				# WKNOTE: from the text locations to token locations
				for idx in range(len(offset_mapping)):
					if (start_idx == -1) & (start < offset_mapping[idx][0]):
						start_idx = idx - 1
					if (end_idx == -1) & (end <= offset_mapping[idx][1]):
						end_idx = idx + 1
				if start_idx == -1:
					start_idx = end_idx
				if (start_idx != -1) & (end_idx != -1):
					label[start_idx:end_idx] = 1
	return torch.tensor(label, dtype=torch.float)


class TrainDataset(Dataset):
	def __init__(self, tokenizer, df):
		self.tokenizer = tokenizer
		self.feature_texts = df['feature_text'].values
		self.pn_historys = df['pn_history'].values
		self.annotation_lengths = df['annotation_length'].values
		self.locations = df['location'].values

	def __len__(self):
		return len(self.feature_texts)

	def __getitem__(self, item):
		inputs = prepare_input(
			self.tokenizer,
			self.pn_historys[item],
			self.feature_texts[item])
		label = create_label(
			self.tokenizer,
			self.pn_historys[item],
			self.annotation_lengths[item],
			self.locations[item])
		return inputs, label