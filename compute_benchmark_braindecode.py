# %% imports
import argparse
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand
import seaborn as sns

import mne
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


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
    age = df_demographics.Age.values
    handcrafted = False

    kind = "eyes-pooled"

elif dataset in ('tuab', 'mmd'):
    kind = 'rest'
    subjects = subjects_df['participant_id']
    age = subjects_df['age']
    handcrafted = False

else:
    raise NotImplementedError

# section above is (almost) a copy of compute_bechmark_handcrafted_features
# ------------------------------------------------------------------------------
# section below is modified to work with braindecode models

# %% Load fif files with preload=False, set age as target, and create dataset
from X_y_model import create_windows_ds
windows_ds, window_size, n_channels = create_windows_ds(fnames, age)

from X_y_model import create_model_and_data_split
n_folds = 10
n_epochs = 3
batch_size = 64
seed = 20211012
metrics = ('neg_mean_absolute_error', 'r2')
for model_name in ['shallow', 'deep']:
    scores = {m: [] for m in metrics}
    for fold_i in range(n_folds):
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        X, y, model = create_model_and_data_split(
            model_name=model_name,
            windows_ds=windows_ds,
            n_channels=n_channels,
            window_size=window_size,
            n_epochs=n_epochs,
            fold=fold_i,
            cv=cv,
            seed=seed,
            batch_size=batch_size,
        )
        model.fit(X=X, y=y, epochs=n_epochs)
        for m in metrics:
            scores[m].append(model.history[-1, ['_'.join(['valid', m])]][0])


# TODO: re-build expected dataframe
"""
        print(f'{score_key}({name}) = {scores.mean()}')
        results.append(pd.DataFrame(this_result))

results = pd.concat(results)

# %% Plot some results
sns.barplot(x='score', y='metric', hue='model',
            data=results.query("metric == 'MAE'"))
plt.savefig('results_braindecode_mae.pdf')
plt.show()
"""
