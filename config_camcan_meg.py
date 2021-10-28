import pathlib

study_name = "age-prediction-benchmark"

bids_root = pathlib.Path(
    '/storage/store/data/camcan/BIDSsep/rest')

deriv_root = pathlib.Path('/storage/store3/derivatives/camcan-bids/derivatives')

subjects_dir = pathlib.Path('/storage/store/data/camcan-mne/freesurfer')

source_info_path_update = {'processing': 'autoreject',
                           'suffix': 'epo'}

inverse_targets = []

noise_cov = 'ad-hoc'

task = 'rest'
sessions = ['rest']  # keep empty for code flow
data_type = 'meg'
ch_types = ['meg']

analyze_channels = ['MEG0111', 'MEG0121', 'MEG0131', 'MEG0141', 'MEG0211',
    'MEG0221', 'MEG0231', 'MEG0241', 'MEG0311', 'MEG0321', 'MEG0331',
    'MEG0341', 'MEG0411', 'MEG0421', 'MEG0431', 'MEG0441', 'MEG0511',
    'MEG0521', 'MEG0531', 'MEG0541', 'MEG0611', 'MEG0621', 'MEG0631',
    'MEG0641', 'MEG0711', 'MEG0721', 'MEG0731', 'MEG0741', 'MEG0811',
    'MEG0821', 'MEG0911', 'MEG0921', 'MEG0931', 'MEG0941', 'MEG1011',
    'MEG1021', 'MEG1031', 'MEG1041', 'MEG1111', 'MEG1121', 'MEG1131',
    'MEG1141', 'MEG1211', 'MEG1221', 'MEG1231', 'MEG1241', 'MEG1311',
    'MEG1321', 'MEG1331', 'MEG1341', 'MEG1411', 'MEG1421', 'MEG1431',
    'MEG1441', 'MEG1511', 'MEG1521', 'MEG1531', 'MEG1541', 'MEG1611',
    'MEG1621', 'MEG1631', 'MEG1641', 'MEG1711', 'MEG1721', 'MEG1731',
    'MEG1741', 'MEG1811', 'MEG1821', 'MEG1831', 'MEG1841', 'MEG1911',
    'MEG1921', 'MEG1931', 'MEG1941', 'MEG2011', 'MEG2021', 'MEG2031',
    'MEG2041', 'MEG2111', 'MEG2121', 'MEG2131', 'MEG2141', 'MEG2211',
    'MEG2221', 'MEG2231', 'MEG2241', 'MEG2311', 'MEG2321', 'MEG2331',
    'MEG2341', 'MEG2411', 'MEG2421', 'MEG2431', 'MEG2441', 'MEG2511',
    'MEG2521', 'MEG2531', 'MEG2541', 'MEG2611', 'MEG2621', 'MEG2631',
    'MEG2641']

l_freq = 0.1
h_freq = 49

eeg_reference = []

eog_channels = []

find_breaks = False

n_proj_eog = 1

reject = None

on_rename_missing_events = "warn"

N_JOBS = 30

decim = 5 # Cam-CAN has 1000 Hz; Cuban Human Brain Project 200Hz

mf_st_duration = 10.
h_freq = None
# XXX the values below differ from our previous papers but would be in line
# with the other EEG data used in this benchmark
epochs_tmin = 0.
epochs_tmax = 10.
rest_epochs_overlap = 0.
rest_epochs_duration = 10.
baseline = None

mf_cal_fname = '/storage/store/data/camcan-mne/Cam-CAN_sss_cal.dat'
mf_ctc_fname = '/storage/store/data/camcan-mne/Cam-CAN_ct_sparse.fif'

find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
run_source_estimation = True
use_template_mri = True

event_repeated = "drop"
l_trans_bandwidth = "auto"

h_trans_bandwidth = "auto"

random_state = 42

shortest_event = 1

log_level = "info"

mne_log_level = "error"

# on_error = 'continue'
on_error = "continue"
