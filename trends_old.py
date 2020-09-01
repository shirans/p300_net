import h5py
import joypy
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nilearn as nl
import nilearn.plotting as nlplt
from dataloaders.trends_kaggle import loading_df, plot_loading

matplotlib.use('MACOSX')
print(matplotlib.get_backend())
print(plt.isinteractive())


def load_icn():
    return pd.read_csv("data/trends_kaggle/ICN_numbers.csv")


def print_missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count() * 100).sort_values(ascending=False)
    missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_train_data.head())


def print_corr(df):
    cols = df.columns[1:]
    fig, ax = plt.subplots(figsize=(20, 20))

    sns.heatmap(df[cols].corr(), annot=True, cmap='RdYlGn', ax=ax)
    plt.show()
    print(df.head(10))
    print(df.mean())
    print(df.head(10)[cols].corr())

    upper = df.where(np.triu(np.ones(df.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.5
    to_drop = [column for column in upper.columns if any(upper[column] > 0.2)]
    print('Very high correlated features: ', to_drop)


def plot_bar(df, feature, title='', show_percent=False, size=2):
    f, ax = plt.subplots(1, 1, figsize=(4 * size, 3 * size))
    total = float(len(df))
    sns.barplot(np.round(df[feature].value_counts().index).astype(int), df[feature].value_counts().values, alpha=0.8,
                palette='Set2')

    plt.title(title)
    if show_percent:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 3,
                    '{:1.2f}%'.format(100 * height / total),
                    ha="center", rotation=45)
    plt.xlabel(feature, fontsize=12, )
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()


# print_missing_data(sc)
# plot_bar(train_data, 'age', 'age count and %age plot', show_percent=True, size=4)


def plot_bar(df, feature, title='', show_percent=False, size=2):
    f, ax = plt.subplots(1, 1, figsize=(4 * size, 3 * size))
    total = float(len(df))
    sns.barplot(np.round(df[feature].value_counts().index).astype(int), df[feature].value_counts().values, alpha=0.8,
                palette='Set2')

    plt.title(title)
    if show_percent:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 3,
                    '{:1.2f}%'.format(100 * height / total),
                    ha="center", rotation=45)
    plt.xlabel(feature, fontsize=12, )
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()


def load_subject(filename, mask_niimg):
    """
    Load a subject saved in .mat format with
        the version 7.3 flag. Return the subject
        niimg, using a mask niimg as a template
        for nifti headers.

    Args:
        filename    <str>            the .mat filename for the subject data
        mask_niimg  niimg object     the mask niimg object used for nifti headers
    """
    subject_data = None
    with h5py.File(subject_filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
    subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)
    return subject_niimg

# icn = load_icn()
#
# print_missing_data(train_data)
# print_missing_data(loading_df)
#
# # Input data files are available in the "../input/" directory.
# # # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# mask_filename = f'{data_path}/fMRI_mask.nii'
# subject_filename = f'{data_path}/fMRI_train/10004.mat'
# smri_filename = '/Users/shiran.s/dev/p300_net/data/trends_kaggle/ch2better.nii'
# mask_niimg = nl.image.load_img(mask_filename)
#
# plot_bar(train_data, 'age', 'age count and %age plot', show_percent=True, size=4)
# features_df = pd.merge(train_data, loading_df, on=['Id'], how='left')
# for col in train_data.columns[2:]:
#     plot_bar(train_data, col, f'{col} count plot', size=4)
# print_corr(features_df)
#
# subject_niimg = load_subject(subject_filename, mask_niimg)
# print("Image shape is %s" % (str(subject_niimg.shape)))
# num_components = subject_niimg.shape[-1]
# print("Detected {num_components} spatial maps".format(num_components=num_components))
# a = nlplt.plot_prob_atlas(subject_niimg, bg_img=smri_filename, view_type='filled_contours', draw_cross=False,
#                           title='All %d spatial maps' % num_components, threshold='auto')
#
# plt.show()
# plt.show()
# nl.plotting.show()
