import importlib
import argparse
from joblib import Parallel, delayed

import pandas as pd

import mne
from mne_bids import BIDSPath
import autoreject

parser = argparse.ArgumentParser(description='Compute autoreject.')
parser.add_argument('-d', '--dataset',
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
task = cfg.task
analyze_channels = cfg.analyze_channels
data_type = cfg.data_type
N_JOBS = cfg.N_JOBS
DEBUG = False

session = None
sessions = cfg.sessions
if dataset in ('tuab', 'camcan'):
    session = sessions[0]

conditions = {
    'lemon': ('eyes/closed', 'eyes/open', 'eyes'),
    'chbp': ('eyes/closed', 'eyes/open', 'eyes'),
    'tuab': ('rest',),
    'camcan': ('rest',)
}[dataset]


subjects_df = pd.read_csv(bids_root / "participants.tsv", sep='\t')

subjects = sorted(sub for sub in subjects_df.participant_id if
                  (deriv_root / sub / session / data_type).exists())
if DEBUG:
    subjects = subjects[:1]
    N_JOBS = 1


def run_subject(subject, task):
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


print(f"computing autorejct on {dataset}")
logging = Parallel(n_jobs=N_JOBS)(
  delayed(run_subject)(sub.split('-')[1], task=task) for sub in subjects)

out_log = pd.DataFrame({"ok": logging, "subject": subjects})
out_log.to_csv(deriv_root / 'autoreject_log.csv')
