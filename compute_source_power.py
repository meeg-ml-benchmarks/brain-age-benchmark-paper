import os.path as op
import pandas as pd

from joblib import Parallel, delayed

import mne
from mne.minimum_norm import apply_inverse_cov
from mne_bids import BIDSPath

from config_camcan_meg import deriv_root

# Paths
fsaverage_subject_dir = "/storage/inria/agramfor/MNE-sample-data/subjects"
subjects_dir = "/storage/store/data/camcan-mne/freesurfer"

# Get the subjects
proc_log = pd.read_csv(deriv_root / 'autoreject_log.csv')
good_subjects = proc_log.query('ok == "OK"').subject
good_subjects

frequency_bands = {
    "low": (0.1, 1),
    "delta": (1, 4),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta_low": (15.0, 26.0),
    "beta_mid": (26.0, 35.0),
    "beta_high": (35.0, 49)
}


def compute_source_power(subject, deriv_root, fsaverage_subject_dir,
                         features, subjects_dir, session='rest',
                         datatype='meg'):

    # Prepare source estimate
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=session,
                         extension='.fif',
                         datatype=datatype,
                         root=deriv_root,
                         check=False)
    fname_info = bids_path.copy().update(processing='clean',
                                         suffix='epo')
    fname_inv = bids_path.copy().update(suffix='inv')
    inv = mne.minimum_norm.read_inverse_operator(fname_inv)
    info = mne.io.read_info(fname_info)
    covs = features['sub-' + subject]['covs']

    # Prepare label time series
    labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub',
                                        subjects_dir=fsaverage_subject_dir)
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
        result[freq_keys[i]] = label_power

    return result


DEBUG = True
N_JOBS = 40
if DEBUG:
    N_JOBS = 1
    good_subjects = good_subjects[:1]

features = mne.externals.h5io.read_hdf5(deriv_root / f'features_rest.h5')
out = Parallel(n_jobs=N_JOBS)(
    delayed(compute_source_power)(subject=subject[4:], deriv_root=deriv_root,
                                  fsaverage_subject_dir=fsaverage_subject_dir,
                                  features=features, subjects_dir=subjects_dir)
    for subject in good_subjects)

mne.externals.h5io.write_hdf5(
    op.join(deriv_root, 'mne_source_power.h5'), out,
    overwrite=True)
