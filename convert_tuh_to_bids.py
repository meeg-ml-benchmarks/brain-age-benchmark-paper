"""Convert TUH Abnormal Dataset to the BIDS format.

See "The Temple University Hospital EEG Corpus: Electrode Location and Channel
Labels" document on TUH EEG's website for more info on the dataset conventions:
https://www.isip.piconepress.com/publications/reports/2020/tuh_eeg/electrodes/

E.g., to run on drago:
>>> python convert_tuh_to_bids.py \
    --tuab_data_dir /storage/store/data/tuh_eeg/www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf \
    --bids_data_dir /storage/store2/data/TUAB-healthy-bids \
    --healthy_only True \
    --reset_session_indices True
"""

import re
import argparse
import datetime

import mne
import numpy as np
from braindecode.datasets import TUHAbnormal
from mne_bids import write_raw_bids, print_dir_tree, make_report, BIDSPath


SEX_TO_MNE = {'n/a': 0, 'm': 1, 'f': 2}


def rename_tuh_channels(ch_name):
    """Rename TUH channels and ignore non-EEG and custom channels.

    Rules:
    - 'Z' should always be lowercase.
    - 'P' following a 'F' should be lowercase.
    """
    exclude = [  # Defined by hand - do we really want to remove them?
        'LOC',
        'ROC',
        'EKG1',
    ]
    match = re.findall(r'^EEG\s([A-Z]\w+)-REF$', ch_name)
    if len(match) == 1:
        out = match[0]
        out = out.replace('FP', 'Fp').replace('Z', 'z')  # Apply rules
    else:
        out = ch_name

    if out in exclude:
        out = ch_name

    return out


def _convert_tuh_recording_to_bids(ds, bids_save_dir, desc=None):
    """Convert single TUH recording to BIDS.

    Parameters
    ----------
    ds : braindecode.datasets.BaseDataset
        TUH recording to convert to BIDS.
    bids_save_dir : st
        Directory where to save the BIDS version of the dataset.
    desc : None | pd.Series
        Description of the recording, containing subject and recording
        information. If None, use `ds.description`.
    """
    raw = ds.raw
    raw.pick_types(eeg=True)  # Only keep EEG channels
    if desc is None:
        desc = ds.description

    # Extract reference
    # XXX Not supported yet in mne-bids: see mne-bids/mne_bids/write.py::766
    ref = re.findall(r'\_tcp\_(\w\w)', desc['path'])
    if len(ref) != 1:
        raise ValueError('Expecting one directory level with tcp in it.')
    elif ref[0] == 'ar':  # average reference
        reference = ''
    elif ref[0] == 'le':  # linked ears
        reference = ''
    else:
        raise ValueError(f'Unknown reference found in file name: {ref[0]}.')

    # Rename channels to a format readable by MNE
    raw.rename_channels(rename_tuh_channels)
    # Ignore channels that are not in the 10-5 system
    montage = mne.channels.make_standard_montage('standard_1005')
    ch_names = np.intersect1d(raw.ch_names, montage.ch_names)
    raw.pick_channels(ch_names)
    raw.set_montage(montage)

    # Add pathology and train/eval labels
    # XXX The following will break if it's not TUAB
    # XXX Also, should be written to the `..._scans.tsv` file instead of being
    #     annotations
    # onset = raw.times[0]
    # duration = raw.times[-1] - raw.times[0]
    # raw.annotations.append(
    #     onset, duration, 'abnormal' if desc['pathological'] else 'normal')
    # raw.annotations.append(
    #     onset, duration, 'train' if desc['train'] else 'eval')

    # Make up birthday based on recording date and age to allow mne-bids to
    # compute age
    birthday = datetime.datetime(desc['year'] - desc['age'], desc['month'], 1)
    birthday -= datetime.timedelta(weeks=4)
    sex = desc['gender'].lower()  # This assumes gender=sex

    # Add additional data required by BIDS
    mrn = str(desc['subject']).zfill(8)  # MRN: Medical Record Number
    session_nb = str(desc['session']).zfill(3)
    subject_info = {
        'participant_id': mrn,
        'birthday': (birthday.year, birthday.month, birthday.day),
        'sex': SEX_TO_MNE[sex],
        'handedness': None  # Not available
    }
    raw.info['line_freq'] = 60  # Data was collected in North America
    raw.info['subject_info'] = subject_info
    task = 'abnormal' if desc['pathological'] else 'normal'

    bids_path = BIDSPath(
        subject=mrn, session=session_nb, task=task, split=desc['segment'],
        root=bids_save_dir, datatype='eeg', check=True)

    write_raw_bids(raw, bids_path, overwrite=True)


def convert_tuab_to_bids(tuh_data_dir, bids_save_dir, healthy_only=True,
                         reset_session_indices=True, concat_split_files=True,
                         n_jobs=1):
    """Convert TUAB dataset to BIDS format.

    Parameters
    ----------
    tuh_data_dir : str
        Directory where the original TUAB dataset is saved, e.g.
        `/tuh_eeg/www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf`.
    bids_save_dir : str
        Directory where to save the BIDS version of the dataset.
    healthy_only : bool
        If True, only convert recordings with "normal" EEG.
    reset_session_indices : bool
        If True, reset session indices so that each subject has a session 001,
        and that there is no gap between session numbers for a subject.
    concat_split_files : bool
        If True, concatenate recordings that were split into a single file.
        This is based on the "token" field of the original TUH file paths.
    n_jobs : None | int
        Number of jobs for parallelization.
    """
    concat_ds = TUHAbnormal(tuh_data_dir, recording_ids=None, n_jobs=n_jobs)

    if healthy_only:
        concat_ds = concat_ds.split(by='pathological')['False']
    description = concat_ds.description  # Make a copy because `description` is
    # made on-the-fly
    if concat_split_files:
        n_segments_per_session = description.groupby(
            ['subject', 'session'])['segment'].apply(list).apply(len)
        if n_segments_per_session.unique() != np.array([1]):
            raise NotImplementedError(
                'Concatenation of split files is not implemented yet.')
        else:
            description['segment'] = '001'

    if reset_session_indices:
        description['session'] = description.groupby(
            'subject')['session'].transform(lambda x: np.arange(len(x)) + 1)

    for ds, (_, desc) in zip(concat_ds.datasets, description.iterrows()):
        assert ds.description['path'] == desc['path']
        _convert_tuh_recording_to_bids(ds, bids_save_dir, desc=desc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert TUH to BIDS.')
    parser.add_argument(
        '--tuab_data_dir', type=str,
        default='/storage/store/data/tuh_eeg/www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf',
        help='Path to the original data.')
    parser.add_argument(
        '--bids_data_dir', type=str,
        help='Path to where the converted data should be saved.')
    parser.add_argument(
        '--healthy_only', type=bool, default=False,
        help='Only convert recordings of healthy subjects (default: False)')
    parser.add_argument(
        '--reset_session_indices', type=bool, default=True,
        help='Reset session indices (default: True)')
    parser.add_argument(
        '--n_jobs', type=int, default=1,
        help='number of parallel processes to use (default: 1)')
    args = parser.parse_args()

    convert_tuab_to_bids(
        args.tuab_data_dir, args.bids_data_dir, healthy_only=args.healthy_only,
        reset_session_indices=args.reset_session_indices, n_jobs=args.n_jobs)

    print_dir_tree(args.bids_data_dir)
    print(make_report(args.bids_data_dir))
