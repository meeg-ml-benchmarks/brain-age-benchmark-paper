# %% imports
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

config_map = {'chbp': "config_chbp_eeg",
              'lemon': "config_lemon_eeg",
              'tuab': "config_tuab",
              'camcan': "config_camcan_meg"}

bench_config = {  # put other benchmark related config here
    'filter_bank': {  # it can go in a seprate file later
        'frequency_bands': {
            "low": (0.1, 1),
            "delta": (1, 4),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 15.0),
            "beta_low": (15.0, 26.0),
            "beta_mid": (26.0, 35.0),
            "beta_high": (35.0, 49)
        }
    }
}

# %% get age

def load_benchmark_data(dataset, benchmark, condition=None):
    """Load the input features and outcome vectors for a given benchmark
    
    Parameters
    ----------
    dataset: 'camcan' | 'chbp' | 'lemon' | 'tuh'
        The input data to consider
    benchmark: 'filter_bank' | 'hand_crafted' | 'deep'
        The input features to consider. If 'deep', no features are loaded.
        Instead information for accsing the epoched data is provided.
    condition: 'eyes-closed' | 'eyes-open' | 'pooled' | 'rest'
        Specify from which sub conditions data should be loaded.
    
    Returns
    -------
    X: numpy.ndarray or pandas.DataFrame of shape(n_subjects, n_predictors)
        The predictors. In the case of the filterbank models, columns can
        represent covariances.
    y: the outcome vector of shape(n_subjects)
        The age used as prediction target.
    model: object
        The model to matching the benachmark-specific features.
        For `filter_bank` and `hand_crafted`, a scikit-learn estimator pipleine
        is returned.  
    """
    if dataset not in config_map:
        raise ValueError(f"We don't know the dataset '{dataset}' you requested.")

    cfg = importlib.import_module(config_map[dataset])
    bids_root = cfg.bids_root
    deriv_root = cfg.deriv_root
    task = cfg.task
    analyze_channels = cfg.analyze_channels

    df_subjects = pd.read_csv(bids_root / "participants.tsv", sep='\t')
    df_subjects = df_subjects.set_index('participant_id')
    # now we read in the processing log to see for which participants we have EEG
    proc_log = pd.read_csv(deriv_root / 'autoreject_log.csv')
    good_subjects = proc_log.query('ok == "OK"').subject

    # handle default for condition.
    if condition is None:
        if dataset in ('chbp', 'lemon'):
            condition_ = 'pooled'
        else:
            condition_ = 'rest'
    else:
        condition_ = condition
            
    df_subjects = df_subjects.loc[good_subjects]
    X, y, model = None, None, None
    if benchmark == 'filterbank-riemann':
        frequency_bands = bench_config['filter_bank']['frequency_bands']
        features = mne.externals.h5io.read_hdf5(
            deriv_root / f'features_{condition_}.h5')
        covs = [features[sub]['covs'] for sub in df_subjects.index]
        covs = np.array(covs)
        X = pd.DataFrame(
            {band: list(covs[:, ii]) for ii, band in
             enumerate(frequency_bands)})
        y = df_subjects.age.values

        filter_bank_transformer = coffeine.make_filter_bank_transformer(
            names=list(frequency_bands),
            method='riemann'
        )
        model = make_pipeline(
            filter_bank_transformer, StandardScaler(),
            RidgeCV(alphas=np.logspace(-5, 10, 100)))

    elif benchmark == 'hand_crafted':
        raise NotImplementedError('not yet available')
    elif benchmark == 'deep':
        raise NotImplementedError('not yet available')

    return X, y, model


#%%
X, y, model = load_benchmark_data(dataset='tuab', benchmark='filterbank-riemann')

dummy_model = DummyRegressor(strategy="median")

models = {
    "filter_bank_model": model,
    "dummy_model": dummy_model
}

# %% Run CV
cv = KFold(n_splits=10, shuffle=True, random_state=42)
results = list()
for metric in ('neg_mean_absolute_error', 'r2'):
    for name, model in models.items():
        scores = cross_val_score(
            model, X_df, y, cv=cv, scoring=metric
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
