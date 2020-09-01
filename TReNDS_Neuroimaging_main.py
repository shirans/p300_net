import h5py
import joypy
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nilearn as nl
import nilearn.plotting as nlplt
from dataloaders.trends_kaggle import loading_df, plot_loading, load_train_scores, plot_train_scores

# main
from run_fc.metadata import Metadata
from run_fc.ready_made_models import train_tree, train_svm

_plot = False
_data_path = "/Users/shiran.s/dev/p300_net/data/trends_kaggle/"

loading_df, targets = loading_df(_data_path)
scores = load_train_scores(_data_path)
ids_train = scores['Id']
tags = scores['age']
load_df_train = loading_df[loading_df.index.isin(ids_train.index)]

from sklearn.model_selection import train_test_split

X = load_df_train.to_numpy()
y = tags.to_numpy()
y = y.round().astype(int)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

train_tree(X_train, y_train, X_valid, y_valid, Metadata(None))
if _plot:
    plot_train_scores(scores)
    plot_loading(loading_df, targets)
