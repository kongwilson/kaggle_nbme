"""
DESCRIPTION

Copyright (C) Weicong Kong, 30/03/2022
"""
import os.path

from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer

from nbme.utils import *

seed_everything(seed=CFG.seed)
LOGGER = get_logger(CFG.hugging_face_model_name)


def preprocess_features(features):
	# turn 'Last-Pap-smear-I-year-ago' to 'Last-Pap-smear-1-year-ago
	features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
	return features


def correct_annotation_in_train_data(train):
	train.loc[338, 'annotation'] = ast.literal_eval('[["father heart attack"]]')
	train.loc[338, 'location'] = ast.literal_eval('[["764 783"]]')

	train.loc[621, 'annotation'] = ast.literal_eval('[["for the last 2-3 months"]]')
	train.loc[621, 'location'] = ast.literal_eval('[["77 100"]]')

	train.loc[655, 'annotation'] = ast.literal_eval('[["no heat intolerance"], ["no cold intolerance"]]')
	train.loc[655, 'location'] = ast.literal_eval('[["285 292;301 312"], ["285 287;296 312"]]')

	train.loc[1262, 'annotation'] = ast.literal_eval('[["mother thyroid problem"]]')
	train.loc[1262, 'location'] = ast.literal_eval('[["551 557;565 580"]]')

	train.loc[1265, 'annotation'] = ast.literal_eval('[[\'felt like he was going to "pass out"\']]')
	train.loc[1265, 'location'] = ast.literal_eval('[["131 135;181 212"]]')

	train.loc[1396, 'annotation'] = ast.literal_eval('[["stool , with no blood"]]')
	train.loc[1396, 'location'] = ast.literal_eval('[["259 280"]]')

	train.loc[1591, 'annotation'] = ast.literal_eval('[["diarrhoe non blooody"]]')
	train.loc[1591, 'location'] = ast.literal_eval('[["176 184;201 212"]]')

	train.loc[1615, 'annotation'] = ast.literal_eval('[["diarrhea for last 2-3 days"]]')
	train.loc[1615, 'location'] = ast.literal_eval('[["249 257;271 288"]]')

	train.loc[1664, 'annotation'] = ast.literal_eval('[["no vaginal discharge"]]')
	train.loc[1664, 'location'] = ast.literal_eval('[["822 824;907 924"]]')

	train.loc[1714, 'annotation'] = ast.literal_eval('[["started about 8-10 hours ago"]]')
	train.loc[1714, 'location'] = ast.literal_eval('[["101 129"]]')

	train.loc[1929, 'annotation'] = ast.literal_eval('[["no blood in the stool"]]')
	train.loc[1929, 'location'] = ast.literal_eval('[["531 539;549 561"]]')

	train.loc[2134, 'annotation'] = ast.literal_eval('[["last sexually active 9 months ago"]]')
	train.loc[2134, 'location'] = ast.literal_eval('[["540 560;581 593"]]')

	train.loc[2191, 'annotation'] = ast.literal_eval('[["right lower quadrant pain"]]')
	train.loc[2191, 'location'] = ast.literal_eval('[["32 57"]]')

	train.loc[2553, 'annotation'] = ast.literal_eval('[["diarrhoea no blood"]]')
	train.loc[2553, 'location'] = ast.literal_eval('[["308 317;376 384"]]')

	train.loc[3124, 'annotation'] = ast.literal_eval('[["sweating"]]')
	train.loc[3124, 'location'] = ast.literal_eval('[["549 557"]]')

	train.loc[3858, 'annotation'] = ast.literal_eval(
		'[["previously as regular"], ["previously eveyr 28-29 days"], ["previously lasting 5 days"], ["previously regular flow"]]')
	train.loc[3858, 'location'] = ast.literal_eval(
		'[["102 123"], ["102 112;125 141"], ["102 112;143 157"], ["102 112;159 171"]]')

	train.loc[4373, 'annotation'] = ast.literal_eval('[["for 2 months"]]')
	train.loc[4373, 'location'] = ast.literal_eval('[["33 45"]]')

	train.loc[4763, 'annotation'] = ast.literal_eval('[["35 year old"]]')
	train.loc[4763, 'location'] = ast.literal_eval('[["5 16"]]')

	train.loc[4782, 'annotation'] = ast.literal_eval('[["darker brown stools"]]')
	train.loc[4782, 'location'] = ast.literal_eval('[["175 194"]]')

	train.loc[4908, 'annotation'] = ast.literal_eval('[["uncle with peptic ulcer"]]')
	train.loc[4908, 'location'] = ast.literal_eval('[["700 723"]]')

	train.loc[6016, 'annotation'] = ast.literal_eval('[["difficulty falling asleep"]]')
	train.loc[6016, 'location'] = ast.literal_eval('[["225 250"]]')

	train.loc[6192, 'annotation'] = ast.literal_eval('[["helps to take care of aging mother and in-laws"]]')
	train.loc[6192, 'location'] = ast.literal_eval('[["197 218;236 260"]]')

	train.loc[6380, 'annotation'] = ast.literal_eval(
		'[["No hair changes"], ["No skin changes"], ["No GI changes"], ["No palpitations"], ["No excessive sweating"]]')
	train.loc[6380, 'location'] = ast.literal_eval(
		'[["480 482;507 519"], ["480 482;499 503;512 519"], ["480 482;521 531"], ["480 482;533 545"], ["480 482;564 582"]]')

	train.loc[6562, 'annotation'] = ast.literal_eval(
		'[["stressed due to taking care of her mother"], ["stressed due to taking care of husbands parents"]]')
	train.loc[6562, 'location'] = ast.literal_eval('[["290 320;327 337"], ["290 320;342 358"]]')

	train.loc[6862, 'annotation'] = ast.literal_eval('[["stressor taking care of many sick family members"]]')
	train.loc[6862, 'location'] = ast.literal_eval('[["288 296;324 363"]]')

	train.loc[7022, 'annotation'] = ast.literal_eval(
		'[["heart started racing and felt numbness for the 1st time in her finger tips"]]')
	train.loc[7022, 'location'] = ast.literal_eval('[["108 182"]]')

	train.loc[7422, 'annotation'] = ast.literal_eval('[["first started 5 yrs"]]')
	train.loc[7422, 'location'] = ast.literal_eval('[["102 121"]]')

	train.loc[8876, 'annotation'] = ast.literal_eval('[["No shortness of breath"]]')
	train.loc[8876, 'location'] = ast.literal_eval('[["481 483;533 552"]]')

	train.loc[9027, 'annotation'] = ast.literal_eval('[["recent URI"], ["nasal stuffines, rhinorrhea, for 3-4 days"]]')
	train.loc[9027, 'location'] = ast.literal_eval('[["92 102"], ["123 164"]]')

	train.loc[9938, 'annotation'] = ast.literal_eval(
		'[["irregularity with her cycles"], ["heavier bleeding"], ["changes her pad every couple hours"]]')
	train.loc[9938, 'location'] = ast.literal_eval('[["89 117"], ["122 138"], ["368 402"]]')

	train.loc[9973, 'annotation'] = ast.literal_eval('[["gaining 10-15 lbs"]]')
	train.loc[9973, 'location'] = ast.literal_eval('[["344 361"]]')

	train.loc[10513, 'annotation'] = ast.literal_eval('[["weight gain"], ["gain of 10-16lbs"]]')
	train.loc[10513, 'location'] = ast.literal_eval('[["600 611"], ["607 623"]]')

	train.loc[11551, 'annotation'] = ast.literal_eval('[["seeing her son knows are not real"]]')
	train.loc[11551, 'location'] = ast.literal_eval('[["386 400;443 461"]]')

	train.loc[11677, 'annotation'] = ast.literal_eval('[["saw him once in the kitchen after he died"]]')
	train.loc[11677, 'location'] = ast.literal_eval('[["160 201"]]')

	train.loc[12124, 'annotation'] = ast.literal_eval('[["tried Ambien but it didnt work"]]')
	train.loc[12124, 'location'] = ast.literal_eval('[["325 337;349 366"]]')

	train.loc[12279, 'annotation'] = ast.literal_eval(
		'[["heard what she described as a party later than evening these things did not actually happen"]]')
	train.loc[12279, 'location'] = ast.literal_eval('[["405 459;488 524"]]')

	train.loc[12289, 'annotation'] = ast.literal_eval(
		'[["experienced seeing her son at the kitchen table these things did not actually happen"]]')
	train.loc[12289, 'location'] = ast.literal_eval('[["353 400;488 524"]]')

	train.loc[13238, 'annotation'] = ast.literal_eval('[["SCRACHY THROAT"], ["RUNNY NOSE"]]')
	train.loc[13238, 'location'] = ast.literal_eval('[["293 307"], ["321 331"]]')

	train.loc[13297, 'annotation'] = ast.literal_eval(
		'[["without improvement when taking tylenol"], ["without improvement when taking ibuprofen"]]')
	train.loc[13297, 'location'] = ast.literal_eval('[["182 221"], ["182 213;225 234"]]')

	train.loc[13299, 'annotation'] = ast.literal_eval('[["yesterday"], ["yesterday"]]')
	train.loc[13299, 'location'] = ast.literal_eval('[["79 88"], ["409 418"]]')

	train.loc[13845, 'annotation'] = ast.literal_eval('[["headache global"], ["headache throughout her head"]]')
	train.loc[13845, 'location'] = ast.literal_eval('[["86 94;230 236"], ["86 94;237 256"]]')

	train.loc[14083, 'annotation'] = ast.literal_eval('[["headache generalized in her head"]]')
	train.loc[14083, 'location'] = ast.literal_eval('[["56 64;156 179"]]')
	return train


def preprocess_annotation(df):
	df['annotation'] = df['annotation'].apply(ast.literal_eval)
	df['location'] = df['location'].apply(ast.literal_eval)
	return df

# ====================================================
# Data Loading
# ====================================================
def load_and_prepare_training_data():
	train = pd.read_csv(TRAIN_PATH)
	# convert the column dtype from str to object (list of strings)
	train = preprocess_annotation(train)

	features = pd.read_csv(FEATURE_PATH)
	features = preprocess_features(features)

	patient_notes = pd.read_csv(PATIENT_NOTES_PATH)

	train = train.merge(features, on=['feature_num', 'case_num'], how='left')
	train = train.merge(patient_notes, on=['pn_num', 'case_num'], how='left')

	train = correct_annotation_in_train_data(train)

	train['annotation_length'] = train['annotation'].apply(len)

	# ====================================================
	# CV split
	# ====================================================
	train_folds_path = os.path.join(DATA_ROOT, 'train_folds.csv')
	if os.path.exists(train_folds_path):
		train = pd.read_csv(train_folds_path)
		train = preprocess_annotation(train)
	else:
		Fold = GroupKFold(n_splits=CFG.n_fold)
		groups = train['pn_num'].values
		for n, (train_index, val_index) in enumerate(Fold.split(train, train['location'], groups)):
			train.loc[val_index, 'fold'] = int(n)
		train['fold'] = train['fold'].astype(int)
		print(train.groupby('fold').size())
		train.to_csv(train_folds_path)

	if CFG.debug:
		print(train.groupby('fold').size())
		train = train.sample(n=1000, random_state=0).reset_index(drop=True)
		print(train.groupby('fold').size())
	return train


# ====================================================
# tokenizer
# ====================================================
def load_tokenizer():
	tokenizer = AutoTokenizer.from_pretrained(CFG.hugging_face_model_name, cache_dir=PRETRAINED_CACHE)
	tokenizer.save_pretrained(
		os.path.join(MODEL_FOLDER, 'tokenizer')
	)
	CFG.tokenizer = tokenizer

# ====================================================
# Define max_len
# ====================================================
def analyse_the_max_len_from_the_training_data():
	tokenizer = CFG.tokenizer
	patient_notes = pd.read_csv(PATIENT_NOTES_PATH)
	features = pd.read_csv(FEATURE_PATH)

	pn_history_lengths = []
	pn_his_col = 'pn_history'
	# WKNOTE: the usage of the tqdm wrapper from tqdm.auto, which wrap a collection with tqdm progress tracking
	pn_collections = tqdm(patient_notes[pn_his_col].fillna("").values, total=len(patient_notes))
	for text in pn_collections:
		length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
		pn_history_lengths.append(length)
	LOGGER.info(f'{pn_his_col} max(lengths): {max(pn_history_lengths)}')

	features_lengths = []
	feature_col = 'feature_text'
	feature_collections = tqdm(features[feature_col].fillna("").values, total=len(features))
	for text in feature_collections:
		length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
		features_lengths.append(length)
	LOGGER.info(f'{feature_col} max(lengths): {max(features_lengths)}')

	# WKNOTE: update the max length setting for the training data using the corresponding backbone hugging face model
	CFG.max_len = max(pn_history_lengths) + max(features_lengths) + 3  # cls & sep & sep
	LOGGER.info(f"max_len: {CFG.max_len}")

