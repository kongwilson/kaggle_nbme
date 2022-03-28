# %% [markdown]
# # Roberta Strikes Back !
# 
# This notebook demonstates that you can reach decent performances with Roberta, if you process predictions correctly.
# 
# The training procedure will not be shared as it would most likely destroy the leaderboard.

# %% [markdown]
# # Initialization

# %% [markdown]
# ### Imports

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-03-03T15:27:39.510177Z","iopub.execute_input":"2022-03-03T15:27:39.510834Z","iopub.status.idle":"2022-03-03T15:27:39.586664Z","shell.execute_reply.started":"2022-03-03T15:27:39.510735Z","shell.execute_reply":"2022-03-03T15:27:39.586014Z"}}
import os
import re
import ast
import json
import glob
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %% [markdown]
# ### Paths

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-03-03T15:27:39.635986Z","iopub.execute_input":"2022-03-03T15:27:39.636179Z","iopub.status.idle":"2022-03-03T15:27:39.639757Z","shell.execute_reply.started":"2022-03-03T15:27:39.636155Z","shell.execute_reply":"2022-03-03T15:27:39.639093Z"}}
DATA_PATH = "../input/nbme-score-clinical-patient-notes/"
OUT_PATH = "../input/nbme-roberta-large/"
WEIGHTS_FOLDER = "../input/nbme-roberta-large/"

NUM_WORKERS = 2

# %% [markdown]
# # Data

# %% [markdown]
# ## Preparation

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-03-03T15:27:39.646154Z","iopub.execute_input":"2022-03-03T15:27:39.646514Z","iopub.status.idle":"2022-03-03T15:27:39.655323Z","shell.execute_reply.started":"2022-03-03T15:27:39.646488Z","shell.execute_reply":"2022-03-03T15:27:39.654476Z"}}
def process_feature_text(text):
    text = re.sub('I-year', '1-year', text)
    text = re.sub('-OR-', " or ", text)
    text = re.sub('-', ' ', text)
    return text


def clean_spaces(txt):
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)
#     txt = re.sub(r'\s+', ' ', txt)
    return txt


def load_and_prepare_test(root=""):
    patient_notes = pd.read_csv(root + "patient_notes.csv")
    features = pd.read_csv(root + "features.csv")
    df = pd.read_csv(root + "test.csv")

    df = df.merge(features, how="left", on=["case_num", "feature_num"])
    df = df.merge(patient_notes, how="left", on=['case_num', 'pn_num'])

    df['pn_history'] = df['pn_history'].apply(lambda x: x.strip())
    df['feature_text'] = df['feature_text'].apply(process_feature_text)

    df['feature_text'] = df['feature_text'].apply(clean_spaces)
    df['clean_text'] = df['pn_history'].apply(clean_spaces)

    df['target'] = ""
    return df

# %% [markdown]
# ### Processing

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-03-03T15:27:39.657146Z","iopub.execute_input":"2022-03-03T15:27:39.657394Z","iopub.status.idle":"2022-03-03T15:27:39.673602Z","shell.execute_reply.started":"2022-03-03T15:27:39.65736Z","shell.execute_reply":"2022-03-03T15:27:39.672908Z"}}
import itertools


def token_pred_to_char_pred(token_pred, offsets):
    char_pred = np.zeros((np.max(offsets), token_pred.shape[1]))
    for i in range(len(token_pred)):
        s, e = int(offsets[i][0]), int(offsets[i][1])  # start, end
        char_pred[s:e] = token_pred[i]

        if token_pred.shape[1] == 3:  # following characters cannot be tagged as start
            s += 1
            char_pred[s: e, 1], char_pred[s: e, 2] = (
                np.max(char_pred[s: e, 1:], 1),
                np.min(char_pred[s: e, 1:], 1),
            )

    return char_pred


def labels_to_sub(labels):
    all_spans = []
    for label in labels:
        indices = np.where(label > 0)[0]
        indices_grouped = [
            list(g) for _, g in itertools.groupby(
                indices, key=lambda n, c=itertools.count(): n - next(c)
            )
        ]

        spans = [f"{min(r)} {max(r) + 1}" for r in indices_grouped]
        all_spans.append(";".join(spans))
    return all_spans


def char_target_to_span(char_target):
    spans = []
    start, end = 0, 0
    for i in range(len(char_target)):
        if char_target[i] == 1 and char_target[i - 1] == 0:
            if end:
                spans.append([start, end])
            start = i
            end = i + 1
        elif char_target[i] == 1:
            end = i + 1
        else:
            if end:
                spans.append([start, end])
            start, end = 0, 0
    return spans

# %% [markdown]
# ## Tokenization

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-03-03T15:27:39.676545Z","iopub.execute_input":"2022-03-03T15:27:39.677014Z","iopub.status.idle":"2022-03-03T15:27:41.387994Z","shell.execute_reply.started":"2022-03-03T15:27:39.67698Z","shell.execute_reply":"2022-03-03T15:27:41.387258Z"}}
import numpy as np
from transformers import AutoTokenizer


def get_tokenizer(name, precompute=False, df=None, folder=None):
    if folder is None:
        tokenizer = AutoTokenizer.from_pretrained(name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(folder)

    tokenizer.name = name
    tokenizer.special_tokens = {
        "sep": tokenizer.sep_token_id,
        "cls": tokenizer.cls_token_id,
        "pad": tokenizer.pad_token_id,
    }

    if precompute:
        tokenizer.precomputed = precompute_tokens(df, tokenizer)
    else:
        tokenizer.precomputed = None

    return tokenizer


def precompute_tokens(df, tokenizer):
    feature_texts = df["feature_text"].unique()

    ids = {}
    offsets = {}

    for feature_text in feature_texts:
        encoding = tokenizer(
            feature_text,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        ids[feature_text] = encoding["input_ids"]
        offsets[feature_text] = encoding["offset_mapping"]

    texts = df["clean_text"].unique()

    for text in texts:
        encoding = tokenizer(
            text,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        ids[text] = encoding["input_ids"]
        offsets[text] = encoding["offset_mapping"]

    return {"ids": ids, "offsets": offsets}


def encodings_from_precomputed(feature_text, text, precomputed, tokenizer, max_len=300):
    tokens = tokenizer.special_tokens

    # Input ids
    if "roberta" in tokenizer.name:
        qa_sep = [tokens["sep"], tokens["sep"]]
    else:
        qa_sep = [tokens["sep"]]

    input_ids = [tokens["cls"]] + precomputed["ids"][feature_text] + qa_sep
    n_question_tokens = len(input_ids)

    input_ids += precomputed["ids"][text]
    input_ids = input_ids[: max_len - 1] + [tokens["sep"]]

    # Token type ids
    if "roberta" not in tokenizer.name:
        token_type_ids = np.ones(len(input_ids))
        token_type_ids[:n_question_tokens] = 0
        token_type_ids = token_type_ids.tolist()
    else:
        token_type_ids = [0] * len(input_ids)

    # Offsets
    offsets = [(0, 0)] * n_question_tokens + precomputed["offsets"][text]
    offsets = offsets[: max_len - 1] + [(0, 0)]

    # Padding
    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([tokens["pad"]] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        offsets = offsets + ([(0, 0)] * padding_length)

    encoding = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "offset_mapping": offsets,
    }

    return encoding


# %% [markdown]
# ## Dataset

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-03-03T15:27:41.391154Z","iopub.execute_input":"2022-03-03T15:27:41.391349Z","iopub.status.idle":"2022-03-03T15:27:41.402216Z","shell.execute_reply.started":"2022-03-03T15:27:41.391324Z","shell.execute_reply":"2022-03-03T15:27:41.401452Z"}}
import torch
import numpy as np
from torch.utils.data import Dataset


class PatientNoteDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer

        self.texts = df['clean_text'].values
        self.feature_text = df['feature_text'].values
        self.char_targets = df['target'].values.tolist()

    def __getitem__(self, idx):
        text = self.texts[idx]
        feature_text = self.feature_text[idx]
        char_target = self.char_targets[idx]

        # Tokenize
        if self.tokenizer.precomputed is None:
            encoding = self.tokenizer(
                feature_text,
                text,
                return_token_type_ids=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
                truncation="only_second",
                max_length=self.max_len,
                padding='max_length',
            )
            raise NotImplementedError("fix issues with question offsets")
        else:
            encoding = encodings_from_precomputed(
                feature_text,
                text,
                self.tokenizer.precomputed,
                self.tokenizer,
                max_len=self.max_len
            )

        return {
            "ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "token_type_ids": torch.tensor(encoding["token_type_ids"], dtype=torch.long),
            "target": torch.tensor([0], dtype=torch.float),
            "offsets": np.array(encoding["offset_mapping"]),
            "text": text,
        }

    def __len__(self):
        return len(self.texts)


# %% [markdown]
# ### Plot predictions

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-03-03T15:27:41.404783Z","iopub.execute_input":"2022-03-03T15:27:41.405345Z","iopub.status.idle":"2022-03-03T15:27:50.10609Z","shell.execute_reply.started":"2022-03-03T15:27:41.405307Z","shell.execute_reply":"2022-03-03T15:27:50.105297Z"}}
import spacy
import numpy as np

def plot_annotation(df, pn_num):
    options = {"colors": {}}

    df_text = df[df["pn_num"] == pn_num].reset_index(drop=True)

    text = df_text["pn_history"][0]
    ents = []

    for spans, feature_text, feature_num in df_text[["span", "feature_text", "feature_num"]].values:
        for s in spans:
            ents.append({"start": int(s[0]), "end": int(s[1]), "label": feature_text})

        options["colors"][feature_text] =  f"rgb{tuple(np.random.randint(100, 255, size=3))}"

    doc = {"text": text, "ents": sorted(ents, key=lambda i: i["start"])}

    spacy.displacy.render(doc, style="ent", options=options, manual=True, jupyter=True)

# %% [markdown]
# # Model

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-03-03T15:27:50.107662Z","iopub.execute_input":"2022-03-03T15:27:50.107918Z","iopub.status.idle":"2022-03-03T15:27:50.126337Z","shell.execute_reply.started":"2022-03-03T15:27:50.107884Z","shell.execute_reply":"2022-03-03T15:27:50.125623Z"}}
import torch
import transformers
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class NERTransformer(nn.Module):
    def __init__(
        self,
        model,
        num_classes=1,
        config_file=None,
        pretrained=True,
    ):
        super().__init__()
        self.name = model
        self.pad_idx = 1 if "roberta" in self.name else 0

        transformers.logging.set_verbosity_error()

        if config_file is None:
            config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        else:
            config = torch.load(config_file)

        if pretrained:
            self.transformer = AutoModel.from_pretrained(model, config=config)
        else:
            self.transformer = AutoModel.from_config(config)

        self.nb_features = config.hidden_size

#         self.cnn = nn.Identity()
        self.logits = nn.Linear(self.nb_features, num_classes)

    def forward(self, tokens, token_type_ids):
        """
        Usual torch forward function

        Arguments:
            tokens {torch tensor} -- Sentence tokens
            token_type_ids {torch tensor} -- Sentence tokens ids
        """
        hidden_states = self.transformer(
            tokens,
            attention_mask=(tokens != self.pad_idx).long(),
            token_type_ids=token_type_ids,
        )[-1]

        features = hidden_states[-1]

        logits = self.logits(features)

        return logits

# %% [markdown]
# # Inference

# %% [markdown]
# ## Loads weights

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-03-03T15:27:50.127743Z","iopub.execute_input":"2022-03-03T15:27:50.128235Z","iopub.status.idle":"2022-03-03T15:27:50.136078Z","shell.execute_reply.started":"2022-03-03T15:27:50.128194Z","shell.execute_reply":"2022-03-03T15:27:50.135423Z"}}
import torch

def load_model_weights(model, filename, verbose=1, cp_folder="", strict=True):
    """
    Loads the weights of a PyTorch model. The exception handles cpu/gpu incompatibilities.

    Args:
        model (torch model): Model to load the weights to.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to load from. Defaults to "".
        strict (bool, optional): Whether to allow missing/additional keys. Defaults to False.

    Returns:
        torch model: Model with loaded weights.
    """
    if verbose:
        print(f"\n -> Loading weights from {os.path.join(cp_folder,filename)}\n")

    try:
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=strict,
        )
    except RuntimeError:
        model.encoder.fc = torch.nn.Linear(model.nb_ft, 1)
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=strict,
        )

    return model

# %% [markdown]
# ## Predict

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-03-03T15:27:50.137313Z","iopub.execute_input":"2022-03-03T15:27:50.137792Z","iopub.status.idle":"2022-03-03T15:27:50.149248Z","shell.execute_reply.started":"2022-03-03T15:27:50.137754Z","shell.execute_reply":"2022-03-03T15:27:50.148455Z"}}
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


def predict(model, dataset, data_config, activation="softmax"):
    """
    Usual predict torch function
    """
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=data_config['val_bs'],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    preds = []
    with torch.no_grad():
        for data in tqdm(loader):
            ids, token_type_ids = data["ids"], data["token_type_ids"]

            y_pred = model(ids.cuda(), token_type_ids.cuda())

            if activation == "sigmoid":
                y_pred = y_pred.sigmoid()
            elif activation == "softmax":
                y_pred = y_pred.softmax(-1)

            preds += [
                token_pred_to_char_pred(y, offsets) for y, offsets
                in zip(y_pred.detach().cpu().numpy(), data["offsets"].numpy())
            ]

    return preds


# %% [markdown]
# ## Inference

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2022-03-03T15:27:50.151658Z","iopub.execute_input":"2022-03-03T15:27:50.152158Z","iopub.status.idle":"2022-03-03T15:27:50.162376Z","shell.execute_reply.started":"2022-03-03T15:27:50.152122Z","shell.execute_reply":"2022-03-03T15:27:50.161539Z"}}
def inference_test(df, exp_folder, config, cfg_folder=None):
    preds = []

    if cfg_folder is not None:
        model_config_file = cfg_folder + config.name.split('/')[-1] + "/config.pth"
        tokenizer_folder = cfg_folder + config.name.split('/')[-1] + "/tokenizers/"
    else:
        model_config_file, tokenizer_folder = None, None

    tokenizer = get_tokenizer(
        config.name, precompute=config.precompute_tokens, df=df, folder=tokenizer_folder
    )

    dataset = PatientNoteDataset(
        df,
        tokenizer,
        max_len=config.max_len,
    )

    model = NERTransformer(
        config.name,
        num_classes=config.num_classes,
        config_file=model_config_file,
        pretrained=False
    ).cuda()
    model.zero_grad()

    weights = sorted(glob.glob(exp_folder + "*.pt"))
    for weight in weights:
        model = load_model_weights(model, weight)

        pred = predict(
            model,
            dataset,
            data_config=config.data_config,
            activation=config.loss_config["activation"]
        )
        preds.append(pred)

    return preds

# %% [markdown]
# # Main

# %% [markdown]
# ## Config

# %% [code] {"execution":{"iopub.status.busy":"2022-03-03T15:27:50.163779Z","iopub.execute_input":"2022-03-03T15:27:50.164045Z","iopub.status.idle":"2022-03-03T15:27:50.17508Z","shell.execute_reply.started":"2022-03-03T15:27:50.164008Z","shell.execute_reply":"2022-03-03T15:27:50.174327Z"}}
class Config:
    # Architecture
    name = "roberta-large"
    num_classes = 1

    # Texts
    max_len = 310
    precompute_tokens = True

    # Training    
    loss_config = {
        "activation": "sigmoid",
    }

    data_config = {
        "val_bs": 16 if "large" in name else 32,
        "pad_token": 1 if "roberta" in name else 0,
    }

    verbose = 1

# %% [markdown]
# ## Data

# %% [code] {"execution":{"iopub.status.busy":"2022-03-03T15:27:50.179024Z","iopub.execute_input":"2022-03-03T15:27:50.179225Z","iopub.status.idle":"2022-03-03T15:27:51.1212Z","shell.execute_reply.started":"2022-03-03T15:27:50.1792Z","shell.execute_reply":"2022-03-03T15:27:51.120511Z"}}
df_test = load_and_prepare_test(root=DATA_PATH)
df_test.head()

# %% [markdown]
# ## Inference

# %% [code] {"execution":{"iopub.status.busy":"2022-03-03T15:27:51.122392Z","iopub.execute_input":"2022-03-03T15:27:51.122671Z","iopub.status.idle":"2022-03-03T15:36:09.199855Z","shell.execute_reply.started":"2022-03-03T15:27:51.122642Z","shell.execute_reply":"2022-03-03T15:36:09.198766Z"}}
preds = inference_test(
    df_test,
    WEIGHTS_FOLDER,
    Config,
    cfg_folder=OUT_PATH
)[0]

# %% [code] {"execution":{"iopub.status.busy":"2022-03-03T15:38:24.715432Z","iopub.execute_input":"2022-03-03T15:38:24.716194Z","iopub.status.idle":"2022-03-03T15:38:25.036669Z","shell.execute_reply.started":"2022-03-03T15:38:24.716156Z","shell.execute_reply":"2022-03-03T15:38:25.035924Z"}}
df_test['preds'] = preds
df_test['preds'] = df_test.apply(lambda x: x['preds'][:len(x['clean_text'])], 1)

df_test['preds'] = df_test['preds'].apply(lambda x: (x > 0.5).flatten())

# %% [markdown]
# ## Plot predictions

# %% [code] {"execution":{"iopub.status.busy":"2022-03-03T15:38:26.645771Z","iopub.execute_input":"2022-03-03T15:38:26.646323Z","iopub.status.idle":"2022-03-03T15:39:13.191209Z","shell.execute_reply.started":"2022-03-03T15:38:26.646284Z","shell.execute_reply":"2022-03-03T15:39:13.190477Z"}}
try:
    df_test['span'] = df_test['preds'].apply(char_target_to_span)
    plot_annotation(df_test, df_test['pn_num'][0])
except:
    pass

# %% [markdown]
# Unlike for deberta, spaces are not included in the offsets, we need to add them manually otherwise this will hurt performances.

# %% [markdown]
# ## Post-processing

# %% [code] {"execution":{"iopub.status.busy":"2022-03-03T15:40:01.858078Z","iopub.execute_input":"2022-03-03T15:40:01.858343Z","iopub.status.idle":"2022-03-03T15:40:01.868163Z","shell.execute_reply.started":"2022-03-03T15:40:01.858312Z","shell.execute_reply":"2022-03-03T15:40:01.86748Z"}}
def post_process_spaces(target, text):
    target = np.copy(target)

    if len(text) > len(target):
        padding = np.zeros(len(text) - len(target))
        target = np.concatenate([target, padding])
    else:
        target = target[:len(text)]

    if text[0] == " ":
        target[0] = 0
    if text[-1] == " ":
        target[-1] = 0

    for i in range(1, len(text) - 1):
        if text[i] == " ":
            if target[i] and not target[i - 1]:  # space before
                target[i] = 0

            if target[i] and not target[i + 1]:  # space after
                target[i] = 0

            if target[i - 1] and target[i + 1]:
                target[i] = 1

    return target

# %% [code] {"execution":{"iopub.status.busy":"2022-03-03T15:40:04.86552Z","iopub.execute_input":"2022-03-03T15:40:04.86623Z","iopub.status.idle":"2022-03-03T15:40:06.809997Z","shell.execute_reply.started":"2022-03-03T15:40:04.866191Z","shell.execute_reply":"2022-03-03T15:40:06.809209Z"}}
df_test['preds_pp'] = df_test.apply(lambda x: post_process_spaces(x['preds'], x['clean_text']), 1)

# %% [code] {"execution":{"iopub.status.busy":"2022-03-03T15:40:11.071676Z","iopub.execute_input":"2022-03-03T15:40:11.071995Z"}}
try:
    df_test['span'] = df_test['preds_pp'].apply(char_target_to_span)
    plot_annotation(df_test, df_test['pn_num'][0])
except:
    pass

# %% [markdown]
# ## Submission

# %% [code]
df_test['location'] = labels_to_sub(df_test['preds_pp'].values)

sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')

sub = sub[['id']].merge(df_test[['id', "location"]], how="left", on="id")

sub.to_csv('submission.csv', index=False)

sub.head()

# %% [markdown]
# Done ! 