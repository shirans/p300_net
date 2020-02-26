from collections import OrderedDict
from random import random, randint
from copy_helpers import load_muse_csv_as_raw__copy, plot_conditions

from mne import Epochs, find_events
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_p300(raw, conditions, event_id, picks=[0, 1, 2, 3], reject={'eeg': 100e-6}, ylim=(-6, 6), title='p300 plot'):
    raw.filter(1, 30, method='iir')

    events = find_events(raw)

    epochs = mne.Epochs(raw, events=events, event_id=event_id,
                        tmin=-0.1, tmax=0.8, baseline=None,
                        reject=reject, preload=True,
                        verbose=False, picks=picks)
    print('sample drop %: ', (1 - len(epochs.events) / len(events)) * 100)

    plot_inner(epochs, conditions, title, ylim)


def plot_inner(epochs, conditions, title, ylim):
    fig, ax = plot_conditions(epochs, conditions=conditions,
                              ci=97.5, n_boot=1000, title=title,
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
