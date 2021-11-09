# %% imports
import argparse
import importlib
from logging import warning

import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
import coffeine

from deep_learning_utils import (
    create_dataset_target_model, get_fif_paths, BraindecodeKFold,
    make_braindecode_scorer, HistoryTracker)


DATASETS = ['chbp', 'lemon', 'tuab', 'camcan']
BENCHMARKS = ['dummy', 'filterbank-riemann', 'filterbank-source', 'handcrafted',
              'shallow', 'deep']
parser = argparse.ArgumentParser(description='Compute features.')
parser.add_argument(
    '-d', '--dataset',
    default=None,
    nargs='+',
    help='the dataset for which features should be computed')
parser.add_argument(
    '-b', '--benchmark',
    default=None,
    nargs='+', help='Type of features to compute')
parser.add_argument(
    '--n_jobs', type=int, default=1,
    help='number of parallel processes to use (default: 1)')

parsed = parser.parse_args()
datasets = parsed.dataset
benchmarks = parsed.benchmark
N_JOBS = parsed.n_jobs
if datasets is None:
    datasets = list(DATASETS)
if benchmarks is None:
    benchmarks = list(BENCHMARKS)
tasks = [(ds, bs) for ds in datasets for bs in benchmarks]
for dataset, benchmark in tasks:
    if dataset not in DATASETS:
        raise ValueError(f"The dataset '{dataset}' passed is unkonwn")
    if benchmark not in BENCHMARKS:
        raise ValueError(f"The benchmark '{benchmark}' passed is unkonwn")
print(f"Running benchmarks: {', '.join(benchmarks)}")
print(f"Datasets: {', '.join(datasets)}")

config_map = {'chbp': "config_chbp_eeg",
              'lemon': "config_lemon_eeg",
              'tuab': "config_tuab_eeg",
              'camcan': "config_camcan_meg"}

bench_config = {  # put other benchmark related config here
    'filterbank-riemann': {  # it can go in a seprate file later
        'frequency_bands': {
            "low": (0.1, 1),
            "delta": (1, 4),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 15.0),
            "beta_low": (15.0, 26.0),
            "beta_mid": (26.0, 35.0),
            "beta_high": (35.0, 49)
        },
        'feature_map': 'fb_covs',
    },
    'filterbank-source':{
        'frequency_bands': {
            "low": (0.1, 1),
            "delta": (1, 4),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 15.0),
            "beta_low": (15.0, 26.0),
            "beta_mid": (26.0, 35.0),
            "beta_high": (35.0, 49)
        },
        'feature_map': 'source_power'},
    'handcrafted': {'feature_map': 'handcrafted'}
}

# %% get age

def aggregate_features(X, func='mean', axis=0):
    aggs = {'mean': np.nanmean, 'median': np.nanmedian}
    return np.vstack([aggs[func](x, axis=axis, keepdims=True) for x in X])

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
    X: numpy.ndarray or pandas.DataFrame of shape (n_subjects, n_predictors)
        The predictors. In the case of the filterbank models, columns can
        represent covariances.
    y: array, shape (n_subjects,)
        The outcome vector containing age used as prediction target.
    model: object
        The model to matching the benchmark-specific features.
        For `filter_bank` and `hand_crafted`, a scikit-learn estimator pipeline
        is returned.
    """
    if dataset not in config_map:
        raise ValueError(
            f"We don't know the dataset '{dataset}' you requested.")

    cfg = importlib.import_module(config_map[dataset])
    bids_root = cfg.bids_root
    deriv_root = cfg.deriv_root
    task = cfg.task
    analyze_channels = cfg.analyze_channels

    # handle default for condition.
    if condition is None:
        if dataset in ('chbp', 'lemon'):
            condition_ = 'pooled'
        else:
            condition_ = 'rest'
    else:
        condition_ = condition
    df_subjects = pd.read_csv(bids_root / "participants.tsv", sep='\t')
    df_subjects = df_subjects.set_index('participant_id')
    # now we read in the processing log to see for which participants we have EEG

    X, y, model = None, None, None
    if benchmark not in ['dummy', 'shallow', 'deep']:
        bench_cfg = bench_config[benchmark]
        feature_label = bench_cfg['feature_map']
        feature_log = f'feature_{feature_label}_{condition_}-log.csv'
        proc_log = pd.read_csv(deriv_root / feature_log)
        good_subjects = proc_log.query('ok == "OK"').subject
        df_subjects = df_subjects.loc[good_subjects]
        print(f"Found data from {len(good_subjects)} subjects")
        if len(good_subjects) == 0:
            return X, y, model

    if benchmark == 'filterbank-riemann':
        frequency_bands = bench_cfg['frequency_bands']
        features = mne.externals.h5io.read_hdf5(
            deriv_root / f'features_{feature_label}_{condition_}.h5')
        covs = [features[sub]['covs'] for sub in df_subjects.index]
        covs = np.array(covs)
        X = pd.DataFrame(
            {band: list(covs[:, ii]) for ii, band in
             enumerate(frequency_bands)})
        y = df_subjects.age.values
        rank = 65 if dataset == 'camcan' else len(analyze_channels) - 1

        filter_bank_transformer = coffeine.make_filter_bank_transformer(
            names=list(frequency_bands),
            method='riemann',
            projection_params=dict(scale='auto', n_compo=rank)
        )
        model = make_pipeline(
            filter_bank_transformer, StandardScaler(),
            RidgeCV(alphas=np.logspace(-5, 10, 100)))

    elif benchmark == 'filterbank-source':
        frequency_bands = bench_cfg['frequency_bands']
        features = mne.externals.h5io.read_hdf5(
            deriv_root / f'features_{feature_label}_{condition_}.h5')
        source_power = [features[sub] for sub in df_subjects.index]
        source_power = np.array(source_power)
        X = pd.DataFrame(
            {band: list(source_power[:,ii])for ii, band in
             enumerate(frequency_bands)})
        y = df_subjects.age.values
        filter_bank_transformer = coffeine.make_filter_bank_transformer(
            names=list(frequency_bands),
            method='log_diag'
        )
        model = make_pipeline(
            filter_bank_transformer, StandardScaler(),
            RidgeCV(alphas=np.logspace(-5, 10, 100)))

    elif benchmark == 'handcrafted':
        features = mne.externals.h5io.read_hdf5(
            deriv_root / f'features_handcrafted_{condition_}.h5')
        X = [features[sub]['feats'] for sub in df_subjects.index]
        y = df_subjects.age.values
        param_grid = {'max_depth': [4, 6, 8, 16, 32, None],
                      'max_features': ['log2', 'sqrt']}
        rf_reg = GridSearchCV(
            RandomForestRegressor(n_estimators=1000,
                                  random_state=42),
            param_grid=param_grid,
            iid=False,
            cv=5)
        model = make_pipeline(
            FunctionTransformer(aggregate_features, kw_args={'func': 'mean'}),
            rf_reg
        )
    elif benchmark == 'dummy':
        y = df_subjects.age.values
        X = np.zeros(shape=(len(y), 1))
        model = DummyRegressor(strategy="mean")

    elif benchmark in ['shallow', 'deep']:
        fif_fnames = get_fif_paths(dataset, cfg)
        ages = df_subjects.age.values
        model_name = benchmark
        n_epochs = 35
        batch_size = 256  # 64
        cropped = True
        seed = 20211022
        reduce_dimensionality = False
        X, y, model = create_dataset_target_model(
            fnames=fif_fnames,
            ages=ages,
            model_name=model_name,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_jobs=N_JOBS,  # use n_jobs for parallel lazy data loading
            cropped=cropped,
            seed=seed,
            debug=True
        )
        # optionally reduce the input dimension of camcan to 65 components
        # as also done for the other benchmarks. use same parameters as in
        # 'filterbank-riemann'
        if dataset == 'camcan' and reduce_dimensionality:
            pipe = make_pipeline(
                coffeine.spatial_filters.ProjCommonSpace(
                    scale='auto', n_compo=65),
                model,
            )
            model = pipe
    return X, y, model

# %% Run CV

def run_benchmark_cv(benchmark, dataset):
    X, y, model = load_benchmark_data(dataset=dataset, benchmark=benchmark)
    if X is None:
        print(
            "no data found for benchmark "
            f"'{benchmark}' on dataset '{dataset}'")
        return

    metrics = [mean_absolute_error, r2_score]
    # cv_params = dict(n_splits=10, shuffle=True, random_state=42)
    cv_params = dict(n_splits=2, shuffle=True, random_state=42)

    if benchmark in ['shallow', 'deep']:
        # turn off most of the mne logging. due to lazy loading we have
        # uncountable logging outputs that do cover the training logging output
        # as well as might slow down code execution
        mne.set_log_level('ERROR')
        # do not run cv in parallel. we assume to only have 1 GPU
        # instead use n_jobs to (lazily) load data in parallel such that the GPU
        # does not have to wait
        if N_JOBS > 1:
            warning('When running deep learning benchmarks joblib can only be '
                    'used to load the data, as cross-validation with n_jobs '
                    'would require one GPU per split.')

        cv = BraindecodeKFold(**cv_params)
        scoring = {m.__name__: make_braindecode_scorer(m) for m in metrics}
        scoring['history'] = HistoryTracker()
    else:
        cv = KFold(**cv_params)
        scoring = {m.__name__: make_scorer(m) for m in metrics}

    print("Running cross validation ...")
    scores = cross_validate(
        model, X, y, cv=cv, scoring=scoring,
        n_jobs=(1 if benchmark in ['filterbank-source', 'shallow', 'deep']
                else N_JOBS))  # XXX too big for joblib
    print("... done.")

    results = pd.DataFrame(
        {'MAE': scores['test_mean_absolute_error'],
         'r2': scores['test_r2_score'],
         'fit_time': scores['fit_time'],
         'score_time': scores['score_time'],
         'dataset': dataset,
         'benchmark': benchmark}
    )
    for metric in ('MAE', 'r2'):
        print(f'{metric}({benchmark}, {dataset}) = {results[metric].mean()}')

    if benchmark in ['shallow', 'deep']:
        history = scoring['history'].to_frame()  # Learning curves
    else:
        history = None

    return results, history


#%% run benchmarks
for dataset, benchmark in tasks:
    print(f"Now running '{benchmark}' on '{dataset}' data")
    results_df, history_df = run_benchmark_cv(benchmark, dataset)
    if results_df is not None:
        results_df.to_csv(
            f"./results/benchmark-{benchmark}_dataset-{dataset}.csv")
    if history_df is not None:
        history_df.to_csv(
            f"./history/benchmark-{benchmark}_dataset-{dataset}.csv")
