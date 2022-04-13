"""
DESCRIPTION

Copyright (C) Weicong Kong, 30/03/2022
"""
import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.models.deberta.tokenization_deberta_fast import DebertaTokenizerFast
from nbme.utils import PRETRAINED_CACHE


# ====================================================
# Model
# ====================================================
class HuggingFaceBackedModel(nn.Module):
	def __init__(self, cfg, config_path=None, pretrained=False):
		super().__init__()
		self.cfg = cfg
		if config_path is None:
			self.config = AutoConfig.from_pretrained(
				cfg.hugging_face_model_name, output_hidden_states=True, cache_dir=PRETRAINED_CACHE)
		else:
			self.config = torch.load(config_path)
		if pretrained:
			self.model = AutoModel.from_pretrained(
				cfg.hugging_face_model_name, config=self.config, cache_dir=PRETRAINED_CACHE)
		else:
			self.model = AutoModel(self.config)
		self.fc_dropout = nn.Dropout(cfg.fc_dropout)
		self.fc = nn.Linear(self.config.hidden_size, 1)
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

