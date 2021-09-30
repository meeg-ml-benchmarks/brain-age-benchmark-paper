from pathlib import Path

study_name = "age-prediction-benchmark"

# On drago
# N_JOBS = 80
# tuab_root = Path("/storage/store/data/tuh_eeg/www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf")
# bids_root = Path( "/storage/store2/data/TUAB-healthy-bids2")
# deriv_root = Path( "/storage/store3/derivatives/TUAB-healthy-bids2")

# On hubert-xps
N_JOBS = 4
tuab_root = Path("/home/hubert/Data/tuh_abnormal")
bids_root = Path("/home/hubert/Data/bids/tuh_abnormal")
deriv_root = Path("/home/hubert/Data/derivatives/tuh_abnormal")

# subjects = ['00002355']

task = "rest"
# task = "normal"

ch_mapping = {
    'EEG FP1-REF': 'Fp1',
    'EEG FP2-REF': 'Fp2',
    'EEG F3-REF': 'F3',
    'EEG F4-REF': 'F4',
    'EEG C3-REF': 'C3',
    'EEG C4-REF': 'C4',
    'EEG P3-REF': 'P3',
    'EEG P4-REF': 'P4',
    'EEG O1-REF': 'O1',
    'EEG O2-REF': 'O2',
    'EEG F7-REF': 'F7',
    'EEG F8-REF': 'F8',
    'EEG T3-REF': 'T3',
    'EEG T4-REF': 'T4',
    'EEG T5-REF': 'T5',
    'EEG T6-REF': 'T6',
    'EEG A1-REF': 'A1',
    'EEG A2-REF': 'A2',
    'EEG FZ-REF': 'Fz',
    'EEG CZ-REF': 'Cz',
    'EEG PZ-REF': 'Pz'
}
analyze_channels = list(ch_mapping.values())
# analyze_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
#                     'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2',
#                     'Fz', 'Cz', 'Pz']

conditions = []

sessions = ["001"]

datatype = "eeg"
ch_types = ["eeg"]

l_freq = 0.1
h_freq = 49

eeg_reference = []

find_breaks = False

spatial_filter = None

reject = None

on_error = "abort"
on_rename_missing_events = "warn"

epochs_tmin = 0
epochs_tmax = 10
rest_epochs_duration = 10.
rest_epochs_overlap = 0.
baseline = None

event_repeated = "drop"
l_trans_bandwidth = "auto"

h_trans_bandwidth = "auto"

random_state = 42

shortest_event = 1

log_level = "info"

mne_log_level = "info"
# on_error = 'continue'
# on_error = "continue"

on_error = 'abort'
# on_error = 'debug'
