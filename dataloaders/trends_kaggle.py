import h5py
import joypy
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nilearn as nl
import nilearn.plotting as nlplt


def plot_loading(loading_df, targets):
    # plt.figure(figsize=(16, 10), dpi=80)
    fig, axes = joypy.joyplot(loading_df, column=list(targets), ylim='own', figsize=(8, 6))

    plt.title('Source-based morphometry loadings distribution', fontsize=22)
    plt.show()

# loading.csv - sMRI SBM loadings for both train and test samples
def loading_df(data_path):
    loading_df = pd.read_csv(data_path + "loading.csv")

    targets = loading_df.columns[1:]
    return loading_df, targets


def load_train_scores(data_path, fill_null=True):
    df = pd.read_csv(data_path + "/train_scores.csv")
    print(df.isna().sum())
    if fill_null:
        df.fillna(df.mean(), inplace=True)
    return df


def plot_train_scores(df):
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    sns.distplot(df['age'], ax=ax[0])
    ax[0].set_title('Age')

    sns.distplot(df['domain1_var1'], ax=ax[1])
    ax[1].set_title('Domain 1 - Var 1')

    sns.distplot(df['domain1_var2'], ax=ax[2])
    ax[2].set_title('Domain 1 - Var 2')


    sns.distplot(df['domain2_var1'], ax=ax[3])
    ax[3].set_title('Domain 2 - Var 1')

    sns.distplot(df['domain2_var2'], ax=ax[4])
    ax[4].set_title('Domain 2 - Var 2')

    fig.suptitle('Target distributions', fontsize=14)
    plt.show()
    print(df.kurtosis())

    return df
