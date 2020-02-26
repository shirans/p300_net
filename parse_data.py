import warnings
from preprocess.helpers import conditions, event_ids
from copy_helpers import load_muse_csv_as_raw__copy, plot_conditions
import pandas as pd
from scipy.io import loadmat

from plot_p300 import plot_p300
from preprocess.preprocess import data_dict_to_df, label_marker, files_to_df, add_marker_choose_columns
from run_fc.train_model import train_svm

warnings.filterwarnings('ignore')
replace_ch_names = None
z_based = [42, 24, 29, 44, 45]
ch_ind = [0, 1, 2, 3]

data_dir_single = '/Users/shiran.s/dev/p300_net/data/IIb/test_loading_code/AAS010R03.mat'
data_dir_valid = '/Users/shiran.s/dev/p300_net/data/IIb/test_loading_code/AAS010R03.mat'
data_dir_new = '/Users/shiran.s/dev/p300_net/data/IIb/train'


# olde code
x_old = loadmat(data_dir_single)
df_old = data_dict_to_df(x_old, z_based)
df_old = df_old.astype(float)
df_old['Marker'] = df_old.apply(lambda row: label_marker(row), axis=1)

# new code
dfs_new = files_to_df(data_dir_single, z_based)
df_new = pd.concat(dfs_new)
df_new = df_new.astype(float)
df_new = add_marker_choose_columns(z_based, df_new)


def save_and_plot(df, title):
    temp_path = "/Users/shiran.s/dev/p300_net/output/train/train.csv"
    df.to_csv(path_or_buf=temp_path, index=False)

    raw = load_muse_csv_as_raw__copy([temp_path], sfreq=240, stim_ind=-1, replace_ch_names=replace_ch_names,
                                     ch_ind=ch_ind)
    plot_p300(raw, conditions, event_ids, picks=ch_ind, reject={'eeg': 100e-4}, ylim=(-100, 100), title=title)


# save_and_plot(df_old, 'old loading')
save_and_plot(df_new, 'new loading')

train_svm("/Users/shiran.s/dev/p300_net/output/train/")
