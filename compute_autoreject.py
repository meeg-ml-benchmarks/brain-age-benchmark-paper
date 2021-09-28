import importlib
import argparse
from joblib import Parallel, delayed

import pandas as pd

import mne
import autoreject

from config_chbp_eeg import bids_root, deriv_root, N_JOBS, analyze_channels

parser = argparse.ArgumentParser(description='Compute autoreject.')
parser.add_argument('-d', '--dataset',
                    help='the dataset for which features should be computed')
args = parser.parse_args()
dataset = args.dataset

config_map = {'chbp': "config_chbp_eeg",
              'lemon': "config_lemon_eeg",
              'tuab': "config_tuab"}
if dataset not in config_map:
    raise ValueError(f"We don't know the dataset '{dataset}' you requested.")

cfg = importlib.import_module(config_map[dataset])
bids_root = cfg.bids_root
deriv_root = cfg.deriv_root
task = cfg.task
analyze_channels = cfg.analyze_channels
N_JOBS = cfg.N_JOBS
DEBUG = False
    

session = ''
sessions = cfg.sessions
if dataset == 'tuab':
    session = f'ses-{sessions[0]}'

conditions = {
    'lemon': ('eyes/closed', 'eyes/open', 'eyes'),
    'chbp': ('eyes/closed', 'eyes/open', 'eyes'),
    'tuab': ('rest',)
}[dataset]


subjects_df = pd.read_csv(bids_root / "participants.tsv", sep='\t')

subjects = sorted(sub for sub in subjects_df.participant_id if
                  (deriv_root / sub / session / 'eeg').exists())
if DEBUG:
    subjects = subjects[:1]
    N_JOBS = 1


def run_subject(subject, task):
    session_code = session + "_" if session else ""
    fname = (deriv_root / subject / session / 'eeg' /
             f'{subject}_{session_code}task-{task}_proc-clean_epo.fif')
    ok = 'OK'
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

    out_fname = str(fname).replace("proc-clean", "proc-clean-pick-ar")
    epochs.save(out_fname, overwrite=True)
    return ok

print(f"computing autorejct on {dataset}")
logging = Parallel(n_jobs=N_JOBS)(
  delayed(run_subject)(sub, task=task) for sub in subjects)

out_log = pd.DataFrame({"ok": logging, "subject": subjects})
out_log.to_csv(deriv_root / 'autoreject_log.csv')
