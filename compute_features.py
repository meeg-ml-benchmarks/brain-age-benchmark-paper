import argparse
import importlib
from multiprocessing import Value

import pandas as pd
from joblib import Parallel, delayed

import mne
import coffeine
from mne_features.feature_extraction import extract_features


parser = argparse.ArgumentParser(description='Compute features.')
parser.add_argument(
    '-d', '--dataset', choices=['chbp', 'lemon', 'tuab', 'camcan'],
    help='the dataset for which features should be computed')
parser.add_argument(
    '-t', '--feature_type', choices=['fb_covs', 'handcrafted'],
    default='fb_covs', help='Type of features to compute')
args = parser.parse_args()
dataset = args.dataset
FEATURE_TYPE = args.feature_type


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
data_type = cfg.datatype
N_JOBS = cfg.N_JOBS
DEBUG = False

conditions = {
    'lemon': ('eyes/closed', 'eyes/open', 'eyes'),
    'chbp': ('eyes/closed', 'eyes/open', 'eyes'),
    'tuab': ('rest',),
    'camcan': ('rest',)
}[dataset]

session = ''
sessions = cfg.sessions
if dataset in ('tuab', 'camcan'):
    session = f'ses-{sessions[0]}'

subjects_df = pd.read_csv(bids_root / "participants.tsv", sep='\t')

subjects = sorted(sub for sub in subjects_df.participant_id if
                  (deriv_root / sub / session / data_type).exists())

frequency_bands = {
    "low": (0.1, 1),
    "delta": (1, 4),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta_low": (15.0, 26.0),
    "beta_mid": (26.0, 35.0),
    "beta_high": (35.0, 49)
}
hc_selected_funcs = [
    'std',
    'rms',
    'kurtosis',
    'skewness',
    'quantile',
    'ptp_amp',
    'mean',
    'pow_freq_bands',
    'spect_entropy',
    'app_entropy',
    'samp_entropy',
    'svd_entropy',
    'hurst_exp',
    'hjorth_complexity',
    'hjorth_mobility',
    'line_length',
    'wavelet_coef_energy',
    'higuchi_fd',
    'zero_crossings',
    'svd_fisher_info',
    'phase_lock_val'
]
hc_func_params = {
    'quantile__q': [0.1, 0.25, 0.75, 0.9],
    'pow_freq_bands__freq_bands': [0, 2, 4, 8, 13, 18, 24, 30, 49],
    'pow_freq_bands__ratios': 'all',
    'pow_freq_bands__ratios_triu': True,
    'pow_freq_bands__log': True,
    'pow_freq_bands__normalize': None,
}

if DEBUG:
    subjects = subjects[:1]
    N_JOBS = 1
    frequency_bands = frequency_bands.pop('alpha')
    hc_selected_funcs = hc_selected_funcs[0]
    hc_func_params = dict()


def extract_fb_covs(epochs, condition):
    features, meta_info = coffeine.compute_features(
        epochs[condition], features=('covs',), n_fft=1024, n_overlap=512,
        fs=epochs.info['sfreq'], fmax=49, frequency_bands=frequency_bands)
    features['meta_info'] = meta_info
    return features


def extract_handcrafted_feats(epochs, condition):
    features = extract_features(
        epochs[condition].get_data(), epochs.info['sfreq'], hc_selected_funcs,
        funcs_params=hc_func_params, n_jobs=1, ch_names=epochs.ch_names,
        reorder_chs=True, return_as_df=False)
    out = {'feats': features}
    return out


def run_subject(subject, task, condition):
    session_code = session + "_" if session else ""
    fname = (deriv_root / subject / session / data_type /
             f'{subject}_{session_code}task-{task}_proc-clean-pick-ar_epo.fif')
    if not fname.exists():
        return 'no file'

    epochs = mne.read_epochs(fname, proj=False)
    if not any(condition in cc for cc in epochs.event_id):
        return 'condition not found'

    try:
        if FEATURE_TYPE == 'fb_covs':
            out = extract_fb_covs(epochs, condition)
        elif FEATURE_TYPE == 'handcrafted':
            out = extract_handcrafted_feats(epochs, condition)
        else:
            NotImplementedError()
    except Exception as err:
        return repr(err)

    return out


for condition in conditions:
    print(f"Computing {FEATURE_TYPE} features on {dataset} for '{condition}'")
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

    out_fname = deriv_root / f'features_{FEATURE_TYPE}_{label}.h5'
    log_out_fname = deriv_root / f'feature_{FEATURE_TYPE}_{label}-log.csv'

    mne.externals.h5io.write_hdf5(
        out_fname,
        out,
        overwrite=True
    )
    print(f'Features saved under {out_fname}.')

    logging = ['OK' if not isinstance(ff, str) else ff for sub, ff in
               zip(subjects, features)]
    out_log = pd.DataFrame({"ok": logging, "subject": subjects})
    out_log.to_csv(log_out_fname)
    print(f'Log saved under {log_out_fname}.')

