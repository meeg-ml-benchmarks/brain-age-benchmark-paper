import pathlib
import mne

study_name = 'age-prediction-benchmark'

bids_root = pathlib.Path(
    "/storage/store2/data/CHBMP_EEG_and_MRI/ds_bids_chbmp/")

deriv_root = pathlib.Path(
    "/storage/store2/derivatives/eeg-pred-modeling-summer-school/")

demo_root = pathlib.Path('/storage/store2/data/CHBMP_Cognitive_Scales')

task = 'protmap'

datatype = 'eeg'
ch_types = ['eeg']

eeg_template_montage = mne.channels.make_standard_montage(
    'standard_1005'
)
eeg_template_montage.rename_channels(
        {'FFT7h': 'FFC7h', 'FFT8h': 'FFC8h'})

l_freq = 0.1
h_freq = 49

eeg_reference = []

eog_channels = ['EOI', 'EOD']

find_breaks = False

spatial_filter = None

reject = None

on_error = 'abort'
on_rename_missing_events = 'warn'

N_JOBS = 30

epochs_tmin = 0
epochs_tmax = 10
baseline = None

rename_events = {
    'artefacto': 'artefact',
    'discontinuity': 'discontinuity',
    'electrodes artifacts': 'artefact',
    'eyes closed': 'eyes/closed',
    'eyes opened': 'eyes/open',
    'fotoestimulacion': 'photic_stimulation',
    'hiperventilacion 1': 'hyperventilation/1',
    'hiperventilacion 2': 'hyperventilation/2',
    'hiperventilacion 3': 'hyperventilation/3',
    'hyperventilation 1': 'hyperventilation/1',
    'hyperventilation 2': 'hyperventilation/2',
    'hyperventilation 3': 'hyperventilation/3',
    'ojos abiertos': 'eyes/closed',
    'ojos cerrados': 'eyes/closed',
    'photic stimulation': 'photic_stimulation',
    'recuperacion': 'recovery',
    'recuperation': 'recovery'
}

conditions = ["eyes/open", "eyes/closed"]

event_repeated = 'drop'
l_trans_bandwidth = 'auto'

h_trans_bandwidth = 'auto'


random_state = 42

shortest_event = 1

log_level = 'info'

mne_log_level = 'error'

# on_error = 'continue'
on_error = 'continue'
