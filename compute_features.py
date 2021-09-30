import argparse
import importlib

import pandas as pd
from joblib import Parallel, delayed

import mne
import coffeine

parser = argparse.ArgumentParser(description='Compute features.')
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
N_JOBS = cfg.N_JOBS
DEBUG = False

conditions = {
    'lemon': ('eyes/closed', 'eyes/open', 'eyes'),
    'chbp': ('eyes/closed', 'eyes/open', 'eyes'),
    'tuab': ('rest',)
}[dataset]

session = ''
sessions = cfg.sessions
if dataset == 'tuab':
    session = f'ses-{sessions[0]}'

subjects_df = pd.read_csv(bids_root / "participants.tsv", sep='\t')

subjects = sorted(sub for sub in subjects_df.participant_id if
                  (deriv_root / sub / session / 'eeg').exists())
if DEBUG:
    subjects = subjects[:1]
    N_JOBS = 1

frequency_bands = {
    "low": (0.1, 1),
    "delta": (1, 4),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta_low": (15.0, 26.0),
    "beta_mid": (26.0, 35.0),
    "beta_high": (35.0, 49)
}


def run_subject(subject, task, condition):
    session_code = session + "_" if session else ""
    fname = (deriv_root / subject / session / 'eeg' /
             f'{subject}_{session_code}task-{task}_proc-clean-pick-ar_epo.fif')
    if not fname.exists():
        return 'no file'

    epochs = mne.read_epochs(fname, proj=False)
    if not any(condition in cc for cc in epochs.event_id):
        return 'condition not found'

    try:
        features = coffeine.compute_features(
            epochs[condition],
            features=('covs',),
            n_fft=1024, n_overlap=512, fs=epochs.info['sfreq'],
            fmax=49, frequency_bands=frequency_bands)
    except Exception as err:
        return repr(err)
    out = {}
    out.update(features[0])
    out['meta_info'] = features[1]
    return out


for condition in conditions:
    print(f"computing features on {dataset} for '{condition}'")
    features = Parallel(n_jobs=N_JOBS)(
        delayed(run_subject)(sub, task=task, condition=condition)
        for sub in subjects)

    out = {sub: ff for sub, ff in zip(subjects, features)
           if not isinstance(ff, str)}

    label = None
    if dataset == "chbp":
        label = 'pooled'
        if '/' in condition:
            label = f'eyes-{condition.split("/")[1]}'
    elif dataset == "tuab":
        label = 'rest'

    out_fname = deriv_root / f'features_{label}.h5'
    log_out_fname = deriv_root / f'feature_{label}-log.csv'

    mne.externals.h5io.write_hdf5(
        out_fname,
        out,
        overwrite=True
    )

    logging = ['OK' if not isinstance(ff, str) else ff for sub, ff in
               zip(subjects, features)]
    out_log = pd.DataFrame({"ok": logging, "subject": subjects})
    out_log.to_csv(log_out_fname)

