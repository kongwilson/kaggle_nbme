# %% [markdown]
# # DeBERTa-v3 Large - 0.883 LB!

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:00:52.413897Z","iopub.execute_input":"2022-03-05T17:00:52.414171Z","iopub.status.idle":"2022-03-05T17:00:52.425865Z","shell.execute_reply.started":"2022-03-05T17:00:52.414139Z","shell.execute_reply":"2022-03-05T17:00:52.425129Z"},"jupyter":{"outputs_hidden":false}}
# The following is necessary if you want to use the fast tokenizer for deberta v2 or v3
# This must be done before importing transformers
import shutil
from pathlib import Path

transformers_path = Path("/opt/conda/lib/python3.7/site-packages/transformers")

input_dir = Path("../input/deberta-v2-3-fast-tokenizer")

convert_file = input_dir / "convert_slow_tokenizer.py"
conversion_path = transformers_path/convert_file.name

if conversion_path.exists():
    conversion_path.unlink()

shutil.copy(convert_file, transformers_path)
deberta_v2_path = transformers_path / "models" / "deberta_v2"

for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py']:
    filepath = deberta_v2_path/filename

    if filepath.exists():
        filepath.unlink()

    shutil.copy(input_dir/filename, filepath)

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:00:55.421592Z","iopub.execute_input":"2022-03-05T17:00:55.421837Z","iopub.status.idle":"2022-03-05T17:00:58.113191Z","shell.execute_reply.started":"2022-03-05T17:00:55.421808Z","shell.execute_reply":"2022-03-05T17:00:58.112468Z"},"jupyter":{"outputs_hidden":false}}
import os
import gc
import ast
import sys
import copy
import json
import math
import string
import pickle
import random
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
# %env TOKENIZERS_PARALLELISM=true

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:00:58.114993Z","iopub.execute_input":"2022-03-05T17:00:58.115487Z","iopub.status.idle":"2022-03-05T17:00:58.121231Z","shell.execute_reply.started":"2022-03-05T17:00:58.11545Z","shell.execute_reply":"2022-03-05T17:00:58.120345Z"},"jupyter":{"outputs_hidden":false}}
class CFG:
    num_workers=4
    path="../input/deberta-v3-large-5-folds-public/"
    config_path=path+'config.pth'
    model="microsoft/deberta-v3-large"
    batch_size=32
    fc_dropout=0.2
    max_len=354
    seed=42
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]

# %% [markdown]
# # Utils

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:01:01.495672Z","iopub.execute_input":"2022-03-05T17:01:01.496387Z","iopub.status.idle":"2022-03-05T17:01:01.506059Z","shell.execute_reply.started":"2022-03-05T17:01:01.496346Z","shell.execute_reply":"2022-03-05T17:01:01.505374Z"},"jupyter":{"outputs_hidden":false}}
def seed_everything(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(seed=CFG.seed)

# %% [markdown]
# # Tokenizer

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:01:03.285501Z","iopub.execute_input":"2022-03-05T17:01:03.286049Z","iopub.status.idle":"2022-03-05T17:01:04.39728Z","shell.execute_reply.started":"2022-03-05T17:01:03.286012Z","shell.execute_reply":"2022-03-05T17:01:04.396543Z"},"jupyter":{"outputs_hidden":false}}
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast

tokenizer = DebertaV2TokenizerFast.from_pretrained('../input/deberta-tokenizer')
CFG.tokenizer = tokenizer

# %% [markdown]
# # Helper functions for scoring

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:01:05.409675Z","iopub.execute_input":"2022-03-05T17:01:05.409921Z","iopub.status.idle":"2022-03-05T17:01:05.419574Z","shell.execute_reply.started":"2022-03-05T17:01:05.409894Z","shell.execute_reply":"2022-03-05T17:01:05.41889Z"},"jupyter":{"outputs_hidden":false}}
# From https://www.kaggle.com/theoviel/evaluation-metric-folds-baseline
def micro_f1(preds, truths):
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)

    return f1_score(truths, preds)


def spans_to_binary(spans, length=None):
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1

    return binary


def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.

    Args:
        preds (list of lists of two ints): Prediction spans.
        truths (list of lists of two ints): Ground truth spans.

    Returns:
        float: f1 score.
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

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:01:06.695933Z","iopub.execute_input":"2022-03-05T17:01:06.696337Z","iopub.status.idle":"2022-03-05T17:01:06.710633Z","shell.execute_reply.started":"2022-03-05T17:01:06.696299Z","shell.execute_reply":"2022-03-05T17:01:06.709957Z"},"jupyter":{"outputs_hidden":false}}
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
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded = tokenizer(text,
                            add_special_tokens=True,
                            return_offsets_mapping=True)
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


def get_score(y_true, y_pred):
    return span_micro_f1(y_true, y_pred)

# %% [markdown]
# # OOF

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:01:08.255672Z","iopub.execute_input":"2022-03-05T17:01:08.256238Z","iopub.status.idle":"2022-03-05T17:01:48.99787Z","shell.execute_reply.started":"2022-03-05T17:01:08.256197Z","shell.execute_reply":"2022-03-05T17:01:48.997073Z"},"jupyter":{"outputs_hidden":false}}
oof = pd.read_pickle(CFG.path+'oof_df.pkl')
truths = create_labels_for_scoring(oof)
char_probs = get_char_probs(oof['pn_history'].values,
                            oof[[i for i in range(CFG.max_len)]].values,
                            CFG.tokenizer)

best_th = 0.5
best_score = 0.
for th in np.arange(0.45, 0.55, 0.01):
    th = np.round(th, 2)
    results = get_results(char_probs, th=th)
    preds = get_predictions(results)
    score = get_score(preds, truths)

    if best_score < score:
        best_th = th
        best_score = score
    print(f"th: {th}  score: {score:.5f}")
print(f"best_th: {best_th}  score: {best_score:.5f}")

# %% [markdown]
# # Data Loading

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:02:06.517743Z","iopub.execute_input":"2022-03-05T17:02:06.518024Z","iopub.status.idle":"2022-03-05T17:02:07.253757Z","shell.execute_reply.started":"2022-03-05T17:02:06.517991Z","shell.execute_reply":"2022-03-05T17:02:07.253032Z"},"jupyter":{"outputs_hidden":false}}
def preprocess_features(features):
    features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    return features


test = pd.read_csv('../input/nbme-score-clinical-patient-notes/test.csv')
submission = pd.read_csv('../input/nbme-score-clinical-patient-notes/sample_submission.csv')
features = pd.read_csv('../input/nbme-score-clinical-patient-notes/features.csv')
patient_notes = pd.read_csv('../input/nbme-score-clinical-patient-notes/patient_notes.csv')

features = preprocess_features(features)

print(f"test.shape: {test.shape}")
display(test.head())
print(f"features.shape: {features.shape}")
display(features.head())
print(f"patient_notes.shape: {patient_notes.shape}")
display(patient_notes.head())

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:02:17.861606Z","iopub.execute_input":"2022-03-05T17:02:17.861865Z","iopub.status.idle":"2022-03-05T17:02:17.89374Z","shell.execute_reply.started":"2022-03-05T17:02:17.861838Z","shell.execute_reply":"2022-03-05T17:02:17.892947Z"},"jupyter":{"outputs_hidden":false}}
test = test.merge(features, on=['feature_num', 'case_num'], how='left')
test = test.merge(patient_notes, on=['pn_num', 'case_num'], how='left')
display(test.head())

# %% [markdown]
# # Dataset

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:02:20.423316Z","iopub.execute_input":"2022-03-05T17:02:20.42391Z","iopub.status.idle":"2022-03-05T17:02:20.431035Z","shell.execute_reply.started":"2022-03-05T17:02:20.42387Z","shell.execute_reply":"2022-03-05T17:02:20.430365Z"},"jupyter":{"outputs_hidden":false}}
def prepare_input(cfg, text, feature_text):
    inputs = cfg.tokenizer(text, feature_text,
                           add_special_tokens=True,
                           max_length=CFG.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)

    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg,
                               self.pn_historys[item],
                               self.feature_texts[item])

        return inputs

# %% [markdown]
# # Model

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:02:22.087753Z","iopub.execute_input":"2022-03-05T17:02:22.088453Z","iopub.status.idle":"2022-03-05T17:02:22.101257Z","shell.execute_reply.started":"2022-03-05T17:02:22.088414Z","shell.execute_reply":"2022-03-05T17:02:22.100058Z"},"jupyter":{"outputs_hidden":false}}
class ScoringModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg

        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.hugging_face_model_name, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.hugging_face_model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
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
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]

        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))

        return output

# %% [markdown]
# # Inference

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:02:25.081838Z","iopub.execute_input":"2022-03-05T17:02:25.082114Z","iopub.status.idle":"2022-03-05T17:02:25.088673Z","shell.execute_reply.started":"2022-03-05T17:02:25.082069Z","shell.execute_reply":"2022-03-05T17:02:25.087935Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:02:26.353577Z","iopub.execute_input":"2022-03-05T17:02:26.35419Z","iopub.status.idle":"2022-03-05T17:04:55.496861Z","shell.execute_reply.started":"2022-03-05T17:02:26.354153Z","shell.execute_reply":"2022-03-05T17:04:55.494344Z"},"jupyter":{"outputs_hidden":false}}
test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(test_dataset,
                         batch_size=CFG.batch_size,
                         shuffle=False,
                         num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
predictions = []
for fold in CFG.trn_fold:
    model = ScoringModel(CFG, config_path=CFG.config_path, pretrained=False)

    state = torch.load(CFG.path+f"{CFG.model.split('/')[1]}_fold{fold}_best.pth",
                           map_location=torch.device('cpu'))

    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, device)
    prediction = prediction.reshape((len(test), CFG.max_len))
    char_probs = get_char_probs(test['pn_history'].values, prediction, CFG.tokenizer)
    predictions.append(char_probs)
    del model, state, prediction, char_probs
    gc.collect()
    torch.cuda.empty_cache()

predictions = np.mean(predictions, axis=0)

# %% [markdown]
# # Submission

# %% [code] {"execution":{"iopub.status.busy":"2022-03-05T17:04:55.498956Z","iopub.execute_input":"2022-03-05T17:04:55.499251Z","iopub.status.idle":"2022-03-05T17:04:55.547274Z","shell.execute_reply.started":"2022-03-05T17:04:55.499211Z","shell.execute_reply":"2022-03-05T17:04:55.540791Z"},"jupyter":{"outputs_hidden":false}}
results = get_results(predictions, th=0.48)
submission['location'] = results
display(submission.head())
submission[['id', 'location']].to_csv('submission.csv', index=False)

# %% [code] {"jupyter":{"outputs_hidden":false}}
