"""Run hyperparameter tuning for deep learning benchmarks.
"""
# %% imports
import argparse
import importlib
from functools import partial

import optuna
import pandas as pd

from deep_learning_utils import create_dataset_target_model, get_fif_paths


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
    seed = 20211022

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_loguniform('batch_size', 8, 1024)
    weight_decay = trial.suggest_loguniform('lr', 1e-5, 1e-1)

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

    # XXX Split the data (avoid data that will be used for testing in the
    #     benchmark)
    # XXX Fit the model and return performance (MAE?)
    # return mae


#%% Run hyperparameter optimization
study = optuna.create_study()
for dataset, benchmark in tasks:
    print(f"Now running '{benchmark}' on '{dataset}' data")
    study.optimize(
        partial(objective, dataset=dataset, benchmark=benchmark), n_trials=100)

    # XXX Save results
