import importlib
import argparse
from joblib import Parallel, delayed

import pandas as pd

import mne
from mne_bids import BIDSPath
import autoreject

DATASETS = ['chbp', 'lemon', 'tuab', 'camcan']
parser = argparse.ArgumentParser(description='Compute autoreject.')
parser.add_argument(
    '-d', '--dataset',
    default=None,
    nargs='+',
    help='the dataset for which preprocessing should be computed')
parser.add_argument(
    '--n_jobs', type=int, default=1,
    help='number of parallel processes to use (default: 1)')
args = parser.parse_args()
dataset = args.dataset
args = parser.parse_args()
datasets = args.dataset
n_jobs = args.n_jobs
if datasets is None:
    datasets = list(DATASETS)
print(f"Datasets: {', '.join(datasets)}")

cfg = importlib.import_module(config_map[dataset])
N_JOBS = (n_jobs if n_jobs else cfg.N_JOBS)
DEBUG = False

if DEBUG:
    subjects = subjects[:1]
    N_JOBS = 1


def prepare_dataset(dataset):
    config_map = {'chbp': "config_chbp_eeg",
                  'lemon': "config_lemon_eeg",
                  'tuab': "config_tuab",
                  'camcan': "config_camcan_meg"}
    if dataset not in config_map:
        raise ValueError(
            f"We don't know the dataset '{dataset}' you requested.")

    cfg = importlib.import_module(config_map[dataset])
    cfg.conditions = {
        'lemon': ('eyes/closed', 'eyes/open', 'eyes'),
        'chbp': ('eyes/closed', 'eyes/open', 'eyes'),
        'tuab': ('rest',),
        'camcan': ('rest',)
    }[dataset]

    cfg.session = None
    sessions = cfg.sessions
    if dataset in ('tuab', 'camcan'):
        cfg.session = sessions[0]

    subjects_df = pd.read_csv(cfg.bids_root / "participants.tsv", sep='\t')
    subjects = sorted(sub for sub in subjects_df.participant_id if
                      (cfg.deriv_root / sub / cfg.session /
                       cfg.data_type).exists())
    return cfg, subjects

def run_subject(subject, cfg):
    deriv_root = cfg.deriv_root
    task = cfg.task
    analyze_channels = cfg.analyze_channels
    data_type = cfg.data_type
    session = cfg.session
    conditions = cfg.conditions

    bp = BIDSPath(root=deriv_root, subject=subject, session=session,
                  datatype=data_type, processing="clean", task=task,
                  check=False, suffix="epo")
    ok = 'OK'
    fname = bp.fpath
    if not fname.exists():
        return 'no file'
    epochs = mne.read_epochs(fname, proj=False)
    has_conditions = any(cond in epochs.event_id for cond in
                         conditions)
    if not has_conditions:
        return 'no event'
    if analyze_channels:
        epochs.pick_channels(analyze_channels)

    ar = autoreject.AutoReject(n_jobs=1, cv=5)
    epochs = ar.fit_transform(epochs)

    bp_out = bp.copy().update(
        processing="autoreject"
    )
    epochs.save(bp_out, overwrite=True)
    return ok

for dataset in datasets:
    cfg, subjects = prepare_dataset(dataset)

    if DEBUG:
        subjects = subjects[:1]
        N_JOBS = 1

    print(f"computing autorejct on {dataset}")
    logging = Parallel(n_jobs=N_JOBS)(
        delayed(run_subject)(sub.split('-')[1], cfg) for sub in subjects)
    out_log = pd.DataFrame({"ok": logging, "subject": subjects})
    out_log.to_csv(cfg.deriv_root / 'autoreject_log.csv')
