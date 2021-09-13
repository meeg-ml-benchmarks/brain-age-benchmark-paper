from joblib import Parallel, delayed

import mne
import pandas as pd

from config_chbp_eeg import bids_root, deriv_root

subjects_list = list(bids_root.glob('*'))

subjects_df = pd.read_csv(bids_root / "participants.tsv", sep='\t')

subjects = [sub for sub in subjects_df.participant_id if
            (deriv_root / sub / 'eeg').exists()]

channels = list()
for sub in subjects:
    fname = (deriv_root / sub / 'eeg' /
             f'{sub}_task-protmap_proc-clean-pick-ar_epo.fif')
    if fname.exists():
        info = mne.io.read_info(fname)
        channels.append(info['ch_names'])

assert all(channels[0] == ch for ch in channels)

df_common_chs = pd.DataFrame({"name": channels[0]})
df_common_chs.to_csv('./outputs/common_channels_validated.csv')
