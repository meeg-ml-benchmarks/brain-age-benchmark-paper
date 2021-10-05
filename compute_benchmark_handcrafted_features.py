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


parser = argparse.ArgumentParser(description='Compute features.')
parser.add_argument(
    '-d', '--dataset', choices=['chbp', 'lemon', 'tuab', 'camcan'],
    help='the dataset for which features should be computed')
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

# %% Load features
features = mne.externals.h5io.read_hdf5(
    deriv_root / f'features_handcrafted_{kind}.h5')

X_feats = [features[sub]['feats'] for sub in subjects]
# X_feats = np.vstack([features[sub]['feats'] for sub in subjects])
# print(X_feats.shape)

y = age

# %% Create models


def aggregate_features(X, func='mean', axis=0):
    aggs = {'mean': np.nanmean, 'median': np.nanmedian}
    return np.vstack([aggs[func](x, axis=axis, keepdims=True) for x in X])


random_forest_model = make_pipeline(
    FunctionTransformer(aggregate_features, kw_args={'func': 'mean'}),
    SimpleImputer(),
    RandomForestClassifier()
)

dummy_model = DummyRegressor(strategy="median")

models = {
    "random_forest_model": random_forest_model,
    "dummy_model": dummy_model
}

# %% Run CV
cv = KFold(n_splits=10, shuffle=True, random_state=42)

results = list()
for metric in ('neg_mean_absolute_error', 'r2'):
    for name, model in models.items():
        scores = cross_val_score(
            model, X_feats, y, cv=cv, scoring=metric, n_jobs=N_JOBS)
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
plt.savefig('results_handcrafted_mae.pdf')
plt.show()
