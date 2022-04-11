"""
DESCRIPTION

Copyright (C) Weicong Kong, 31/03/2022
"""
import pandas as pd

from nbme.utils import *
from nbme.preprocess import *

LOGGER = get_logger(CFG.hugging_face_model_name)


if __name__ == '__main__':
	train = load_and_prepare_training_data()
	load_tokenizer()
	if CFG.train:
		oof_df = pd.DataFrame()
		for fold in range(CFG.n_fold):
			if fold in CFG.train_folds:
				_oof_df, best_score = train_loop(train, fold)
				oof_df = pd.concat([oof_df, _oof_df])
				LOGGER.info(f"========== fold: {fold} result ==========")
				get_result(_oof_df)
				if best_score < 0.81:
					# WK: since deberta-base can achieve 0.86 on avg, so if any model can't beat that, not shortlisted
					break

		oof_df = oof_df.reset_index(drop=True)
		LOGGER.info(f"========== CV ==========")
		get_result(oof_df)
		oof_df.to_pickle(os.path.join(MODEL_FOLDER, 'oof_df.pkl'))  # WK: save the oof

	if CFG.wandb:
		wandb.finish()
