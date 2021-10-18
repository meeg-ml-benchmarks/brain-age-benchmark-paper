# %% imports
import argparse
import importlib
import numpy as np
import pandas as pd

import mne
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score


parser = argparse.ArgumentParser(description='Braindecode network decoding.')
parser.add_argument(
    '-d', '--dataset', choices=['chbp', 'lemon', 'tuab', 'camcan'],
    help='the dataset to be used in braindecode network decoding')
args = parser.parse_args()
dataset = args.dataset

config_map = {'chbp': "config_chbp_eeg",
              'lemon': "config_lemon_eeg",
              'tuab': "config_tuab",
              'camcan': "config_camcan_meg"}
if dataset not in config_map:
    raise ValueError(f"We don't know the dataset '{dataset}' you requested.")

cfg = importlib.import_module(config_map[dataset])
bids_root = cfg.bids_root
deriv_root = cfg.deriv_root
N_JOBS = cfg.N_JOBS

subjects_df = pd.read_csv(bids_root / "participants.tsv", sep='\t')

# %% get age

# now we read in the processing log to see for which participants we have EEG
proc_log = pd.read_csv(deriv_root / 'autoreject_log.csv')
good_subjects = proc_log.query('ok == "OK"').subject
good_subjects

if dataset == 'chbp':
    df_demographics = pd.read_csv(
        bids_root / '..' / '..' / 'CHBMP_Cognitive_Scales' /
        'Demographic_data.csv', header=1
    )

    df_demographics = df_demographics.iloc[:, :5].set_index('Code')
    df_demographics.index = "sub-" + df_demographics.index

    # Restrict to good subjects
    df_demographics = df_demographics.loc[good_subjects]
    subjects = df_demographics.index
    ages = df_demographics.Age.values
    handcrafted = False

    kind = "eyes-pooled"

elif dataset in ('tuab', 'mmd'):
    kind = 'rest'
    subjects = subjects_df['participant_id']
    ages = subjects_df['age']
    handcrafted = False

else:
    raise NotImplementedError

# section above is (almost) a copy of compute_bechmark_handcrafted_features
# ------------------------------------------------------------------------------
# section below is modified to work with braindecode models

# TODO: implement missing link between above and below

# code below expects to get a list of .fif file names 'fnames' pointing to
# epoched data as well as a list of ages 'ages'
from X_y_model import get_estimator_and_X, cross_val_score
n_folds = 10
cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
n_epochs = 35
batch_size = 64
seed = 20211012

for model_name in ['shallow', 'deep']:
    estimator, X = get_estimator_and_X(
        fnames=fnames,
        ages=ages,
        model_name=model_name,
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    scores = cross_val_score(
        estimator=estimator,
        X=X,
        cv=cv,
        fit_params={'epochs': n_epochs},
    )
    print(model_name, scores)
