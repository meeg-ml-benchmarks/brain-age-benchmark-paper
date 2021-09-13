import pandas as pd

import mne
from config_chbp_eeg import bids_root, deriv_root

subjects_list = list(bids_root.glob('*'))

subjects_df = pd.read_csv(bids_root / "participants.tsv", sep='\t')

subjects = [sub for sub in subjects_df.participant_id if
            (deriv_root / sub / 'eeg').exists()]

eeg_template_montage = mne.channels.make_standard_montage(
    'standard_1005'
)
eeg_template_montage.rename_channels(
    {'FFT7h': 'FFC7h', 'FFT8h': 'FFC8h'})

channels = list()
for sub in subjects:
    fname = deriv_root / sub / 'eeg' / f'{sub}_task-protmap_epo.fif'
    if fname.exists():
        info = mne.io.read_info(fname)
        channels.append(info['ch_names'])

chan_template = eeg_template_montage.ch_names
common_channels = set(chan_template).intersection(*channels)

df_common_names = pd.DataFrame({"name": sorted(common_channels)})
df_common_names.to_csv('./common_channels.csv')
