import argparse
from multiprocessing import Value

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mne
from mne_bids import BIDSPath
import coffeine
from mne_features.feature_extraction import extract_features
from mne.minimum_norm import apply_inverse_cov

from utils import prepare_dataset

DATASETS = ['chbp', 'lemon', 'tuab', 'camcan']
FEATURE_TYPE = ['fb_covs', 'handcrafted', 'source_power']
parser = argparse.ArgumentParser(description='Compute features.')
parser.add_argument(
    '-d', '--dataset',
    default=None,
    nargs='+',
    help='the dataset for which features should be computed')
parser.add_argument(
    '-t', '--feature_type',
    default=None,
    nargs='+', help='Type of features to compute')
parser.add_argument(
    '--n_jobs', type=int, default=1,
    help='number of parallel processes to use (default: 1)')

args = parser.parse_args()
datasets = args.dataset
feature_types = args.feature_type
n_jobs = args.n_jobs
if datasets is None:
    datasets = list(DATASETS)
if feature_types is None:
    feature_types = list(FEATURE_TYPE)
tasks = [(ds, bs) for ds in datasets for bs in feature_types]
for dataset, feature_type in tasks:
    if dataset not in DATASETS:
        raise ValueError(f"The dataset '{dataset}' passed is unkonwn")
    if feature_type not in FEATURE_TYPE:
        raise ValueError(f"The benchmark '{feature_type}' passed is unkonwn")
print(f"Running benchmarks: {', '.join(feature_types)}")
print(f"Datasets: {', '.join(datasets)}")
DEBUG = False
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
    'svd_fisher_info'
]
hc_func_params = {
    'quantile__q': [0.1, 0.25, 0.75, 0.9],
    'pow_freq_bands__freq_bands': [0, 2, 4, 8, 13, 18, 24, 30, 49],
    'pow_freq_bands__ratios': 'all',
    'pow_freq_bands__ratios_triu': True,
    'pow_freq_bands__log': True,
    'pow_freq_bands__normalize': None,
}

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
        return_as_df=False)
    out = {'feats': features}
    return out


def extract_source_power(bp, info, subject, subjects_dir, covs):
    fname_inv = bp.copy().update(suffix='inv',
                                 processing=None,
                                 extension='.fif')
    inv = mne.minimum_norm.read_inverse_operator(fname_inv)
    # Prepare label time series
    labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub',
                                        subjects_dir=subjects_dir)
    labels = mne.morph_labels(
        labels, subject_from='fsaverage', subject_to=subject,
        subjects_dir=subjects_dir)

    labels = [ll for ll in labels if 'unknown' not in ll.name]

    # for each frequency band
    result = []
    for i in range(covs.shape[0]):
        cov = mne.Covariance(data=covs[i, :, :],
                             names=info['ch_names'],
                             bads=info['bads'],
                             projs=info['projs'],
                             nfree=0)  # nfree ?
        stc = apply_inverse_cov(cov, info, inv,
                                lambda2=1. / 9.,
                                pick_ori='normal',
                                nave=1,
                                method="MNE")

        label_power = mne.extract_label_time_course(stc,
                                                    labels,
                                                    inv['src'],
                                                    mode="mean")
        result.append(np.diag(label_power[:,0]))
    return result


def run_subject(subject, cfg, condition):
    task = cfg.task
    deriv_root = cfg.deriv_root
    data_type = cfg.data_type
    session = cfg.session
    if session.startswith('ses-'):
        session = session.lstrip('ses-')

    bp_args = dict(root=deriv_root, subject=subject,
                   datatype=data_type, processing="autoreject",
                   task=task,
                   check=False, suffix="epo")
    if session:
        bp_args['session'] = session
    bp = BIDSPath(**bp_args)

    if not bp.fpath.exists():
        return 'no file'

    epochs = mne.read_epochs(bp, proj=False, preload=True)
    if not any(condition in cc for cc in epochs.event_id):
        return 'condition not found'
    out = None
    # make sure that no EOG/ECG made it into the selection
    epochs.pick_types(**{data_type: True})
    try:
        if feature_type == 'fb_covs':
            out = extract_fb_covs(epochs, condition)
        elif feature_type == 'handcrafted':
            out = extract_handcrafted_feats(epochs, condition)
        elif feature_type == 'source_power':
            covs = extract_fb_covs(epochs, condition)
            covs = covs['covs']
            out = extract_source_power(
                bp, epochs.info, subject, cfg.subjects_dir, covs)
        else:
            NotImplementedError()
    except Exception as err:
        return repr(err)

    return out


for dataset, feature_type in tasks:
    cfg, subjects = prepare_dataset(dataset)
    N_JOBS = cfg.N_JOBS if not n_jobs else n_jobs
    if DEBUG:
        subjects = subjects[:1]
        N_JOBS = 1
        frequency_bands = {"alpha": (8.0, 15.0)}
        hc_selected_funcs = ['std']
        hc_func_params = dict()

    for condition in cfg.feature_conditions:
        print(
            f"Computing {feature_type} features on {dataset} for '{condition}'")
        features = Parallel(n_jobs=N_JOBS)(
            delayed(run_subject)(sub.split('-')[1], cfg=cfg,
            condition=condition) for sub in subjects)

        out = {sub: ff for sub, ff in zip(subjects, features)
               if not isinstance(ff, str)}

        label = None
        if dataset in ("chbp", "lemon"):
            label = 'pooled'
            if '/' in condition:
                label = f'eyes-{condition.split("/")[1]}'
        elif dataset in ("tuab", 'camcan'):
            label = 'rest'

        out_fname = cfg.deriv_root / f'features_{feature_type}_{label}.h5'
        log_out_fname = (
            cfg.deriv_root / f'feature_{feature_type}_{label}-log.csv')

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
