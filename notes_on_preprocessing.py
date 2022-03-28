"""
DESCRIPTION

Copyright (C) Weicong Kong, 25/03/2022
"""

from nbme.utils import *
import pandas as pd
import ast

train = pd.read_csv(TRAIN_PATH)

# annotation is given as str, should turn them into objects (in this case, list of strings)
annotation = train.iloc[0]['annotation']
print(type(annotation))
# ast.literal_eval is just like eval() function which evaluates a string of Python code,
#   but not as dangerous as eval() because it just considers a small subset of Python's syntax to be valid:
annotation = ast.literal_eval(annotation)
print(type(annotation))

# same for location
location = train.iloc[0]['location']
print(location, type(location))
location = ast.literal_eval(location)
print(location, type(location))

# How many cases are in the dataset
print('Total num of cases is', train['case_num'].nunique())
# one case have quite a few patient notes
# each patient note has labels for all features identified for the case




