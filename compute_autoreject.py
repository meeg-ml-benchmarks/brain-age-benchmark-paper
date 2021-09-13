from joblib import Parallel, delayed

import pandas as pd

import mne
import autoreject

from config_chbp_eeg import bids_root, deriv_root, N_JOBS, analyze_channels

subjects_list = list(bids_root.glob('*'))

subjects_df = pd.read_csv(bids_root / "participants.tsv", sep='\t')

subjects = [sub for sub in subjects_df.participant_id if
            (deriv_root / sub / 'eeg').exists()]

# df_common_names = pd.read_csv('./outputs/common_channels.csv')


def run_subject(subject):
    fname = (deriv_root / subject / 'eeg' /
             f'{subject}_task-protmap_proc-clean_epo.fif')
    ok = 'OK'
    if not fname.exists():
        return 'no file'
    epochs = mne.read_epochs(fname, proj=False)
    has_eyes = any(eyes in epochs.event_id for eyes in
                   ('eyes/closed', 'eyes/open'))
    if not has_eyes:
        return 'no event'
    # epochs = epochs['eyes']
    # epochs.pick_channels(df_common_names.name.values)
    epochs.pick_channels(analyze_channels)

    if True:
        ar = autoreject.AutoReject(n_jobs=1, cv=5)
        epochs = ar.fit_transform(epochs)

    if False:
        reject = autoreject.get_rejection_threshold(
            epochs=epochs, ch_types=['eeg'], decim=1,
            verbose=False
        )
        epochs.drop_bad(reject)

    out_fname = deriv_root / subject / \
        'eeg' / f'{subject}_task-protmap_proc-clean-pick-ar_epo.fif'

    # epochs.set_eeg_reference('average')
    epochs.save(out_fname, overwrite=True)
    return ok


logging = Parallel(n_jobs=N_JOBS)(
  delayed(run_subject)(sub) for sub in subjects)

out_log = pd.DataFrame({"ok": logging, "subject": subjects})
out_log.to_csv(deriv_root / 'autoreject_log.csv')
