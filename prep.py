"""
DESCRIPTION

Copyright (C) Weicong Kong, 19/03/2022
"""
import os
import spacy.displacy
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
import seaborn as sns
from holoviews import annotate

pd.options.display.max_columns = 50
pd.options.display.width = 500


DATA_ROOT = os.path.join('nbme-score-clinical-patient-notes')

TRAIN_PATH = os.path.join(DATA_ROOT, 'train.csv')
TEST_PATH = os.path.join(DATA_ROOT, 'test.csv')
FEATURE_PATH = os.path.join(DATA_ROOT, 'features.csv')
PATIENT_NOTES_PATH = os.path.join(DATA_ROOT, 'patient_notes.csv')

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
features = pd.read_csv(FEATURE_PATH)
patient_notes = pd.read_csv(PATIENT_NOTES_PATH)


def annotate_sample(note_num):
	note_num = int(note_num)
	warnings.filterwarnings('ignore')
	patient_df = train[train["pn_num"] == note_num].copy()
	patient_df = patient_df.merge(features[['feature_num', 'feature_text']], on='feature_num')
	# WK: location should be a list of str, which some ";" should be handled and turned to ","
	patient_df["location"] = patient_df["location"].str.replace("'", '"').str.replace(';', '","').apply(json.loads)
	annotation = patient_df["feature_text"]
	ents = []
	for idx, row in patient_df.iterrows():
		spans = row['location']
		label = row['feature_text']
		for span in spans:
			start_loc = span.split()[0]
			end_loc = span.split()[1]
			ents.append({
				'start': int(start_loc),
				'end': int(end_loc),
				'label': label
			})
	doc = {
		'text': patient_notes[patient_notes["pn_num"] == note_num]["pn_history"].iloc[0],
		"ents": ents
	}
	p1 = sns.color_palette('hls', annotation.nunique(), desat=1).as_hex()
	p2 = sns.color_palette('hls', annotation.nunique(), desat=0.5).as_hex()
	colors = {k: f"linear-gradient(90deg, {c1}, {c2})" for k, c1, c2 in zip(annotation.unique(), p1, p2)}
	options = {"colors": colors}
	html = spacy.displacy.render(doc, style="ent", options=options, manual=True, jupyter=False)
	return html


sample_html = annotate_sample(16)
with open(os.path.join('data', 'sample_vis.html'), 'w') as f:
	f.write(sample_html)
