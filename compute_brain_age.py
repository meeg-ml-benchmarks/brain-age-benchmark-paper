# %% imports
import argparse
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mne
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

import coffeine


parser = argparse.ArgumentParser(description='Compute autoreject.')
parser.add_argument('-d', '--dataset',
                    help='the dataset for which features should be computed')
args = parser.parse_args()
dataset = args.dataset

config_map = {'chbp': "config_chbp_eeg", 'tuab': "config_tuab"}
if dataset not in config_map:
    raise ValueError(f"We don't know the dataset '{dataset}' you requested.")

cfg = importlib.import_module(config_map[dataset])
bids_root = cfg.bids_root
deriv_root = cfg.deriv_root
task = cfg.task
analyze_channels = cfg.analyze_channels
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

    kind = "eyes-pooled"

elif dataset == 'tuab':
    kind = 'rest'
    subjects = subjects_df['participant_id']
    age = subjects_df['age']

else:
    raise NotImplementedError

features = mne.externals.h5io.read_hdf5(deriv_root / f'features_{kind}.h5')
covs = [features[sub]['covs'] for sub in subjects]
X_covs = np.array(covs)
print(X_covs.shape)

frequency_bands = {
    "low": (0.1, 1),
    "delta": (1, 4),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta_low": (15.0, 26.0),
    "beta_mid": (26.0, 35.0),
    "beta_high": (35.0, 49)
}

# %% Get X and y
X_df = pd.DataFrame(
  {band: list(X_covs[:, ii]) for ii, band in enumerate(frequency_bands)})
y = age

# %% Create models

filter_bank_transformer = coffeine.make_filter_bank_transformer(
    names=list(frequency_bands),
    method='riemann'
)

filter_bank_model = make_pipeline(filter_bank_transformer, StandardScaler(),
                                  RidgeCV(alphas=np.logspace(-5, 10, 100)))

dummy_model = DummyRegressor(strategy="median")

models = {
    "filter_bank_model": filter_bank_model,
    "dummy_model": dummy_model
}

# %% Run CV
cv = KFold(n_splits=10, shuffle=True, random_state=42)

results = list()
for metric in ('neg_mean_absolute_error', 'r2'):
    for name, model in models.items():
        scores = cross_val_score(
            model, X_df, y, cv=cv, scoring=metric, n_jobs=N_JOBS
        )
        score_key = metric
        if metric == 'neg_mean_absolute_error':
            score_key = "MAE"
            scores *= -1

        this_result = {"metric": score_key,
                       "score": scores,
                       "model": name}

        print(f'{score_key}({name}) = {scores.mean()}')
        results.append(pd.DataFrame(this_result))

results = pd.concat(results)

# %% Plot some results
sns.barplot(x='score', y='metric', hue='model',
            data=results.query("metric == 'MAE'"))
plt.savefig(f'results_mae_{dataset}.pdf')
plt.show()
