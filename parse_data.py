from collections import OrderedDict
from random import random, randint

from mne import Epochs, find_events
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from copy_helpers import load_muse_csv_as_raw__copy, plot_conditions

from scipy.io import loadmat

warnings.filterwarnings('ignore')


def plot_p300(raw, conditions, event_id, picks = [0,1,2,3], reject={'eeg': 100e-6}, ylim=(-6, 6)):
    raw.filter(1, 30, method='iir')

    events = find_events(raw)

    epochs = mne.Epochs(raw, events=events, event_id=event_id,
                    tmin=-0.1, tmax=0.8, baseline=None,
                    reject=reject, preload=True,
                    verbose=False, picks=picks)
    print('sample drop %: ', (1 - len(epochs.events) / len(events)) * 100)

    plt.show()
    fig, ax = plot_conditions(epochs, conditions=conditions,
                              ci=97.5, n_boot=1000, title='',
                              diff_waveform=(1, 2), ylim=ylim)

    plt.show()


def load_orig():
    # load orig file
    f_p300 = "/Users/shiran.s/dev/eeg-notebooks/data/visual/P300/subject1/session1/data_2017-02-04-15_45_13.csv"
    p300 = pd.read_csv(f_p300)
    raw_orig = load_muse_csv_as_raw__copy([f_p300], 256)
    #
    conditions = OrderedDict()
    conditions['Non-target'] = [1]
    conditions['Target'] = [2]
    #
    event_id_orig = {'Non-Target': 1, 'Target': 2}
    plot_p300(raw_orig, conditions, event_id_orig)



x = loadmat('/Users/shiran.s/dev/p300_net/data/IIb/AAS010R02.mat')

for k in x.keys():
    if isinstance(x[k], bytes) or isinstance(x[k], str) or isinstance(x[k], list):
        print("{} : {} , {}".format(k, len(x[k]), type(x[k])))
    else:
        print("{} : {}".format(k, x[k].shape))

for k in range(0, 10):
    print("runnr:{}, trial:{}, sample:{}, signal:{} StimulusType:{}".format(
        x['runnr'][k], x['trialnr'][k], x['samplenr'][k], x['signal'][k].shape, x['StimulusType'][k]))
print("------------")
runnr = x['runnr']
trialnr = x['trialnr']
Flashing = x['Flashing']

# samplenr = x['samplenr'][0:10][:, 0]
# signal = x['signal'][0:10]
# StimulusType = x['StimulusType'][0:10]
samplenr = x['samplenr'][:, 0]
# samplenr = samplenr + 1486223115
signal = x['signal']
StimulusType = x['StimulusType']
StimulusCode = x['StimulusCode']
trialnr = x['trialnr']


# choose sepcific columns
z_based = [42, 24, 29, 44, 45]
z_based = [5, 6, 7, 8, 9]
one_based = [43, 25, 30, 45, 46]
signal = signal[:, z_based]

sig = np.column_stack((samplenr, signal, StimulusCode, StimulusType, trialnr))
df = pd.DataFrame(sig)

def label_marker(row):
    if row[6] == 0:
        return 0  # no event
    if row[6] > 0 and row[7] == 1:
        return 2  # target
    return 1  # non target

#
# def label_marker(row):
#     st_code = row[6]
#     if st_code == 0:
#         return 0  # no event
#     st_type = row[7]
#     a = 2 if st_code > 0 and st_type else 1
#     # return a
#     if st_code %2 ==1:
#         return a
#     b = 1 if a ==2 else 2
#     return b

# df[['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']] = df[['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']].astype(float)
df = df.astype(float)
df['Marker'] = df.apply (lambda row: label_marker(row), axis=1)


# df = df.rename(columns={0: "timestamps", 1: "TP9", 2: "AF7", 3: "AF8", 4: "TP10", 5: "Right AUX"})
# df = df[['timestamps','TP9', 'AF7', 'AF8', 'TP10', 'Right AUX','Marker']]

# df = df[[0,1,2, 3, 4, 'Marker']]
indexes_to_include = list(range(0, 7))
df = df[indexes_to_include + ['Marker']]
# df = df.rename(columns={0: "timestamps", 1: "TP9", 2: "AF7", 3: "AF8", 4: "TP10"})
replace_ch_names= None

# use all 64 chammels
replace_ch_names = {'1': "TP9"}
print(df.head(10))
ch_ind =indexes_to_include[:-1]



temp_path = "/Users/shiran.s/dev/p300_net/output/temp_csv.csv"
df.to_csv(path_or_buf=temp_path, index=False)
conditions = OrderedDict()
conditions['Non-target'] = [1]
conditions['Target'] = [2]
event_ids = {'Non-Target': 1, 'Target': 2}

raw = load_muse_csv_as_raw__copy([temp_path], sfreq=240, stim_ind=-1, replace_ch_names=replace_ch_names, ch_ind=ch_ind)
plot_p300(raw, conditions, event_ids, picks=ch_ind,reject={'eeg': 100e-4}, ylim=(-100, 100))

# kaggle comp

# file_name = "/Users/shiran.s/dev/p300_net_old/data/train/Data_S02_Sess01.csv"
# sub2_sess1 = pd.read_csv(file_name)
# labels = pd.read_csv("/Users/shiran.s/dev/p300_net_old/data/TrainLabels.csv")


# sub2_sess1 = sub2_sess1.rename(columns={"Time": "timestamps", "FeedBackEvent": "Marker"})

# print(" subject 2 session")
# print("{}".format(sub2_sess1.groupby("FeedBackEvent").count()))
# print("labels")
# print("{}".format(labels))
