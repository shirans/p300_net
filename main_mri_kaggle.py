import joypy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_train_scores():
    df = pd.read_csv("data/trends_kaggle/train_scores.csv")

    # print(round(df.isna().sum() / len(df) * 100, 2))
    df.fillna(df.mean(), inplace=True)
    # print(df.isna().sum())
    #
    # fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    # sns.distplot(df['age'], ax=ax[0])
    # ax[0].set_title('Age')
    #
    # sns.distplot(df['domain1_var1'], ax=ax[1])
    # ax[1].set_title('Domain 1 - Var 1')
    #
    # sns.distplot(df['domain1_var2'], ax=ax[2])
    # ax[2].set_title('Domain 1 - Var 2')
    #
    #
    # sns.distplot(df['domain2_var1'], ax=ax[3])
    # ax[3].set_title('Domain 2 - Var 1')
    #
    # sns.distplot(df['domain2_var2'], ax=ax[4])
    # ax[4].set_title('Domain 2 - Var 2')
    #
    # fig.suptitle('Target distributions', fontsize=14)
    # plt.show()
    # print(df.kurtosis())

    return df


def load_icn():
    return pd.read_csv("data/trends_kaggle/ICN_numbers.csv")

def loading_df():
    loading_df = pd.read_csv("data/trends_kaggle/loading.csv")

    targets = loading_df.columns[1:]

    # plt.figure(figsize=(16, 10), dpi=80)
    # fig, axes = joypy.joyplot(loading_df, column=list(targets), ylim='own', figsize=(14, 10))
    #
    # plt.title('Source-based morphometry loadings distribution', fontsize=22)
    # plt.show()
    return loading_df, targets


def print_corr(df):
    cols = df.columns[1:]
    fig, ax = plt.subplots(figsize=(20, 20))

    sns.heatmap(features_df[cols].corr(), annot=True, cmap='RdYlGn', ax=ax)
    plt.show()
    print(df.head(10))
    print(df.mean())
    print(df.head(10)[cols].corr())


features_df = load_train_scores()
icn = load_icn()
loading_df, targets = loading_df()

features_df = pd.merge(features_df, loading_df, on=['Id'], how='left')
print_corr(features_df)

#

