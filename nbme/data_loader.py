"""
DESCRIPTION

Copyright (C) Weicong Kong, 30/03/2022
"""
import numpy as np
import torch
from torch.utils.data import Dataset

# from nbme.utils import CFG


# WKNOTE: the preproccessing of preparing model INPUTS and LABELS
# WKNOTE: an input is a [SEP] concatenated text, with the first part being the pn_history,
#   and the second part being the feature text
def prepare_input(tokenizer, text, feature_text, max_len):
	inputs = tokenizer(
		text, feature_text,
		add_special_tokens=True,
		max_length=max_len,
		padding="max_length",
		return_offsets_mapping=True,
		return_overflowing_tokens=True,
		truncation=True
	)
	for k, v in inputs.items():
		if k == 'input_ids':
			inputs[k] = torch.tensor(v, dtype=torch.long)
		else:
			inputs[k] = torch.tensor(v, dtype=torch.short)
	return inputs


# WKNOTE: a label is a 1-d tensor of `max_len`, with the label texts being 1, others being 0, and [pad] being -1
#   if `max_len` is shorter than the text tokens, the text will be truncated into multiple pieces
def create_label(tokenizer, text, annotation_length, location_list, max_len):
	encoded = tokenizer(
		text,
		add_special_tokens=True,
		max_length=max_len,
		padding="max_length",
		return_offsets_mapping=True,
		return_overflowing_tokens=True,
		truncation=True
	)
	offset_mapping = encoded['offset_mapping']
	# WKNOTE: encoded.sequence_ids(batch_index: int = 0) -
	#   Return a list mapping the tokens to the id of their original sentences
	#       - `None` for special tokens added around or between sequences
	#       - `0` for tokens corresponding to words in the first sequence
	#       - `1` for tokens corresponding to words in the second sequence when a pair of sequences was jointly encoded
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
	def __init__(self, tokenizer, df, max_len):
		self.tokenizer = tokenizer
		self.max_len = max_len
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
			self.feature_texts[item], self.max_len)
		label = create_label(
			self.tokenizer,
			self.pn_historys[item],
			self.annotation_lengths[item],
			self.locations[item], self.max_len)
		return inputs, label
