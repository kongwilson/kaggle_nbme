"""
DESCRIPTION

Copyright (C) Weicong Kong, 21/03/2022
"""
import os


DATA_ROOT = os.path.join('nbme-score-clinical-patient-notes')

TRAIN_PATH = os.path.join(DATA_ROOT, 'train.csv')
TEST_PATH = os.path.join(DATA_ROOT, 'test.csv')
FEATURE_PATH = os.path.join(DATA_ROOT, 'features.csv')
PATIENT_NOTES_PATH = os.path.join(DATA_ROOT, 'patient_notes.csv')

OUTPUT_DIR = os.path.join('output')
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR, exist_ok=True)