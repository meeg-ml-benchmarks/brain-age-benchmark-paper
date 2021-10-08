from sys import path
from pathlib import Path

study_name = "age-prediction-benchmark"

# On drago
N_JOBS = 30
bids_root = Path("/storage/store2/data/TUAB-healthy-bids2")
deriv_root = Path("/storage/store3/derivatives/TUAB-healthy-bids2")

# subjects = ['00002355']

task = "rest"
# task = "normal"

analyze_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
                    'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2',
                    'Fz', 'Cz', 'Pz']

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
