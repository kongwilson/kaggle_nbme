"""
DESCRIPTION

Copyright (C) Weicong Kong, 31/03/2022
"""
import pandas as pd

import wandb

from nbme.utils import *
from nbme.data_loader import ProjectDataLoader
from nbme.model import HuggingFaceBackedModel, Trainer


# TODO: handle CUDA out of memory by reducing the max_len of the model
# TODO: further re-organise the code so that it can be easily transfer to a cloud env.


# WK: the global configuration var
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

# WK: define model saving paths
MODEL_STORE = os.path.join('model_store')
PRETRAINED_CACHE = os.path.join('cache')

LOGGER = get_logger(CFG.hugging_face_model_name)


seed_everything(seed=CFG.seed)

if __name__ == '__main__':

	loader = ProjectDataLoader(DATA_ROOT, TRAIN_PATH, TEST_PATH, FEATURE_PATH, PATIENT_NOTES_PATH)
	trainer = Trainer(CFG, loader, model_store_dir=MODEL_STORE, pretrained_cache_dir=PRETRAINED_CACHE)
	train_data = loader.load_and_prepare_training_data(CFG.n_fold, debug=CFG.debug)
	trainer.analyse_the_max_len_from_the_training_data(train_data)
	if CFG.train:
		oof_df = pd.DataFrame()
		for fold in range(CFG.n_fold):
			if fold in CFG.train_folds:
				_oof_df, best_score = trainer.train_loop(train_data, fold, HuggingFaceBackedModel)
				oof_df = pd.concat([oof_df, _oof_df])
				LOGGER.info(f"========== fold: {fold} result ==========")
				trainer.get_result(_oof_df)
				if best_score < 0.81:
					# WK: since deberta-base can achieve 0.86 on avg, so if any model can't beat that, not shortlisted
					break

		oof_df = oof_df.reset_index(drop=True)
		LOGGER.info(f"========== CV ==========")
		trainer.get_result(oof_df)
		oof_df.to_pickle(os.path.join(trainer.model_folder, 'oof_df.pkl'))  # WK: save the oof

	if CFG.wandb:
		wandb.finish()
