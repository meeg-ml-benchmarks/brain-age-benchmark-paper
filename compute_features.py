import argparse
import importlib
from multiprocessing import Value
from types import SimpleNamespace

import pandas as pd
from joblib import Parallel, delayed

import mne
from mne_bids import BIDSPath
import coffeine
from mne_features.feature_extraction import extract_features
from mne.minimum_norm import apply_inverse_cov

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


def extract_source_power(bp, subject, subjects_dir, covs):
    info = mne.io.read_info(bp)
    fname_inv = bp.copy().update(suffix='inv')
    inv = mne.minimum_norm.read_inverse_operator(fname_inv)
    # Prepare label time series
    labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub',
                                        subjects_dir=subjects_dir)
    labels = mne.morph_labels(labels,
                              subject_from='fsaverage',
                              subject_to=subject,
                              subjects_dir=subjects_dir)
    labels = [ll for ll in labels if 'unknown' not in ll.name]

    # for each frequency band
    result = dict()
    freq_keys = frequency_bands.keys()
    for i in range(covs.shape[0]):
        cov = mne.Covariance(data=covs[i, :, :],
                             names=info['ch_names'],
                             bads=info['bads'],
                             projs=info['projs'],
                             nfree=0)  # nfree ?
        stc = apply_inverse_cov(cov, info, inv,
                                nave=1,
                                method="dSPM")

        label_power = mne.extract_label_time_course(stc,
                                                    labels,
                                                    inv['src'],
                                                    mode="mean")
        result[freq_keys[i]] = label_power  # needs to be transform in diag matrix

    return result    


def prepare_dataset(dataset):
    config_map = {'chbp': "config_chbp_eeg",
                  'lemon': "config_lemon_eeg",
                  'tuab': "config_tuab",
                  'camcan': "config_camcan_meg"}
    if dataset not in config_map:
        raise ValueError(
            f"We don't know the dataset '{dataset}' you requested.")

    cfg_in = importlib.import_module(config_map[dataset])
    cfg_out = SimpleNamespace(
        bids_root=cfg_in.bids_root,
        deriv_root=cfg_in.deriv_root,
        task=cfg_in.task,
        analyze_channels=cfg_in.analyze_channels,
        data_type=cfg_in.data_type,
        subjects_dir=cfg_in.subjects_dir
    )
    cfg_out.conditions = {
        'lemon': ('eyes/closed', 'eyes/open', 'eyes'),
        'chbp': ('eyes/closed', 'eyes/open', 'eyes'),
        'tuab': ('rest',),
        'camcan': ('rest',)
    }[dataset]

    cfg_out.session = ''
    sessions = cfg_in.sessions
    if dataset in ('tuab', 'camcan'):
        cfg_out.session = 'ses-' + sessions[0]

    subjects_df = pd.read_csv(cfg_out.bids_root / "participants.tsv", sep='\t')
    subjects = sorted(sub for sub in subjects_df.participant_id if
                      (cfg_out.deriv_root / sub / cfg_out.session /
                       cfg_out.data_type).exists())
    return cfg_out, subjects


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

    epochs = mne.read_epochs(bp, proj=False)
    if not any(condition in cc for cc in epochs.event_id):
        return 'condition not found'

    try:
        if feature_type == 'fb_covs':
            out = extract_fb_covs(epochs, condition)
        elif feature_type == 'handcrafted':
            out = extract_handcrafted_feats(epochs, condition)
        elif feature_type == 'source_power':  # needs that fb_covs are already extracted
            label = None
            if '/' in condition:
                label = f'eyes-{condition.split("/")[1]}'
            elif condition == 'rest':
                label = 'rest'
            else:
                label = 'pooled'
            covs_path = deriv_root / f'features_fb_covs_{label}.h5'
            if covs_path.exists():
                covs = mne.externals.h5io.read_hdf5(covs_path)
                covs = features['sub-' + subject]['covs']
                out = extract_source_power(bp, subject, cfg.subjects_dir, covs)
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

    for condition in cfg.conditions:
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
