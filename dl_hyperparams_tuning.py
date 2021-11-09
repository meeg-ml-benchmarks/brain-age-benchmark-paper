"""Run hyperparameter tuning for deep learning benchmarks.
"""
# %% imports
import argparse
import importlib
from functools import partial

import optuna
import pandas as pd
from skorch.helper import SliceDataset
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
import coffeine

from deep_learning_utils import (create_dataset_target_model, get_fif_paths,
                                 predict_recordings, BraindecodeTrainValidSplit)


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


def objective(trial, dataset, benchmark):
    """Load the input features and outcome vectors for a given benchmark

    Parameters
    ----------
    dataset: 'camcan' | 'chbp' | 'lemon' | 'tuh'
        The input data to consider
    benchmark: 'filter_bank' | 'hand_crafted' | 'deep'
        The input features to consider. If 'deep', no features are loaded.
        Instead information for accsing the epoched data is provided.

    Returns
    -------

    """
    if dataset not in config_map:
        raise ValueError(
            f"We don't know the dataset '{dataset}' you requested.")

    cfg = importlib.import_module(config_map[dataset])
    bids_root = cfg.bids_root
    df_subjects = pd.read_csv(bids_root / "participants.tsv", sep='\t')
    df_subjects = df_subjects.set_index('participant_id')
    # now we read in the processing log to see for which participants we have EEG

    fif_fnames = get_fif_paths(dataset, cfg)
    ages = df_subjects.age.values
    model_name = benchmark
    n_epochs = 35
    # XXX Add patience hyperparameter
    seed = 20211022

    reduce_dimensionality = trial.suggest_categorical(
        'reduce_dimensionality', [True, False])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    # batch_size = trial.suggest_loguniform('batch_size', 8, 1024)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)

    X, y, model = create_dataset_target_model(
        fnames=fif_fnames,
        ages=ages,
        model_name=model_name,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_jobs=N_JOBS,  # use n_jobs for parallel lazy data loading
        seed=seed,
        lr=lr,
        weight_decay=weight_decay,
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
    # make an 80/20 split. make sure to use a different seed than the one in the
    # benchmark to avoid accidentally ending up with optimizing one of the
    # benchmark splits
    ds_train, ds_valid = BraindecodeTrainValidSplit(
        .2, random_state=20211109)(X, y)
    # fit the model
    model.fit(SliceDataset(ds_train, idx=0), SliceDataset(ds_train, idx=1))
    # compute the performance on recording level
    y_true, y_pred = predict_recordings(
        model, SliceDataset(ds_valid, idx=0), SliceDataset(ds_valid, idx=1))
    return mean_absolute_error(y_true, y_pred)


#%% Run hyperparameter optimization
study = optuna.create_study()
for dataset, benchmark in tasks:
    print(f"Now running '{benchmark}' on '{dataset}' data")
    study.optimize(
        partial(objective, dataset=dataset, benchmark=benchmark), n_trials=100)

    # Save results
    df = study.trials_dataframe()
    df.to_csv(f"./HPO/benchmark-{benchmark}_dataset-{dataset}.csv")
