import pandas as pd
from joblib import Parallel, delayed

import mne
import coffeine

from config_chbp_eeg import bids_root, deriv_root, N_JOBS

subjects_df = pd.read_csv(bids_root / "participants.tsv", sep='\t')

subjects = [sub for sub in subjects_df.participant_id if
            (deriv_root / sub / 'eeg').exists()]

frequency_bands = {
    "low": (0.1, 1),
    "delta": (1, 4),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta_low": (15.0, 26.0),
    "beta_mid": (26.0, 35.0),
    "beta_high": (35.0, 49)
}


def run_subject(subject, condition):
    fname = (deriv_root / subject / 'eeg' /
             f'{subject}_task-protmap_proc-clean-pick-ar_epo.fif')
    if not fname.exists():
        return 'no file'

    epochs = mne.read_epochs(fname, proj=False)

    features = coffeine.compute_features(
        epochs[condition],
        n_fft=1024, n_overlap=512, fs=epochs.info['sfreq'],
        fmax=49, frequency_bands=frequency_bands)
    out = {}
    out.update(features[0])
    out['meta_info'] = features[1]
    return out


for condition in ('eyes/closed', 'eyes/open', 'eyes'):
    features = Parallel(n_jobs=N_JOBS)(
        delayed(run_subject)(sub, condition=condition)
        for sub in subjects)

    out = {sub: ff for sub, ff in zip(subjects, features)
           if not isinstance(ff, str)}

    label = 'pooled'
    if '/' in condition:
        _, label = condition.split("/")

    mne.externals.h5io.write_hdf5(
        deriv_root / f'features_eyes-{label}.h5',
        out,
        overwrite=True
    )

    logging = ['OK' if not isinstance(ff, str) else ff for sub, ff in
               zip(subjects, features)]
    out_log = pd.DataFrame({"ok": logging, "subject": subjects})
    out_log.to_csv(deriv_root / f'feature_eyes-{label}-log.csv')
