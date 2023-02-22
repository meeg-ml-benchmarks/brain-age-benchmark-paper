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
import glob
import os

from joblib import Parallel, delayed
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from mne_bids import write_raw_bids, print_dir_tree, make_report, BIDSPath

from mne.io.edf.edf import _get_info
from braindecode.datasets.base import BaseDataset, BaseConcatDataset


SEX_TO_MNE = {'n/a': 0, 'm': 1, 'f': 2}


mne.set_log_level('warning')


def _read_edf_header(file_path):
    f = open(file_path, "rb")
    header = f.read(88)
    f.close()
    return header


def _parse_age_and_gender_from_edf_header(file_path):
    header = _read_edf_header(file_path)
    # bytes 8 to 88 contain ascii local patient identification
    # see https://www.teuniz.net/edfbrowser/edf%20format%20description.html
    patient_id = header[8:].decode("ascii")
    age = -1
    found_age = re.findall(r"Age:(\d+)", patient_id)
    if len(found_age) == 1:
        age = int(found_age[0])
    gender = "X"
    found_gender = re.findall(r"\s([F|M])\s", patient_id)
    if len(found_gender) == 1:
        gender = found_gender[0]
    return age, gender


def _parse_description_from_file_path(file_path):
    # stackoverflow.com/questions/3167154/how-to-split-a-dos-path-into-its-components-in-python  # noqa
    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    # version 3.0
    # expect file paths as file_type/split/status/reference/aaaaaaav_s004_t000.edf
    #                      edf/train/normal/01_tcp_ar/aaaaaaav_s004_t000.edf

    version = 'V3.0'
    info, *_ = _get_info(
        file_path, stim_channel='auto', eog=None, 
        misc=None, exclude=(), infer_types=False, preload=False)
    date = info['meas_date']
    fname = tokens[-1].replace('.edf', '')
    subject_id, session, segment = fname.split('_')
    return {
        'path': file_path,
        'version': version,
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'subject': subject_id,  # V3.0 has no longer subject numbers
        'session': int(session[1:]),
        'segment': int(segment[1:]),
    }

def _create_chronological_description(file_paths):
    # this is the first loop (fast)
    descriptions = []
    for file_path in file_paths:
        description = _parse_description_from_file_path(file_path)
        descriptions.append(pd.Series(description))
    descriptions = pd.concat(descriptions, axis=1)
    # order descriptions chronologically

    descriptions.sort_values(
        ["subject", "session", "segment", "year", "month", "day"],
        axis=1, inplace=True)
    # https://stackoverflow.com/questions/42284617/reset-column-index-pandas
    descriptions = descriptions.T.reset_index(drop=True).T
    return descriptions

class TUH(BaseConcatDataset):
    """Temple University Hospital (TUH) EEG Corpus
    (www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tueg).
    Parameters
    ----------
    path: str
        Parent directory of the dataset.
    recording_ids: list(int) | int
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    target_name: str
        Can be 'gender', or 'age'.
    preload: bool
        If True, preload the data of the Raw objects.
    add_physician_reports: bool
        If True, the physician reports will be read from disk and added to the
        description.
    n_jobs: int
        Number of jobs to be used to read files in parallel.
    """
    def __init__(self, path, recording_ids=None, target_name=None,
                 preload=False, add_physician_reports=False, n_jobs=1):
        # create an index of all files and gather easily accessible info
        # without actually touching the files
        file_paths = glob.glob(os.path.join(path, '**/*.edf'), recursive=True)
        descriptions = _create_chronological_description(file_paths)
        # limit to specified recording ids before doing slow stuff
        if recording_ids is not None:
            descriptions = descriptions[recording_ids]
        # this is the second loop (slow)
        # create datasets gathering more info about the files touching them
        # reading the raws and potentially preloading the data
        # disable joblib for tests. mocking seems to fail otherwise
        if n_jobs == 1:
            base_datasets = [self._create_dataset(
                descriptions[i], target_name, preload, add_physician_reports)
                for i in descriptions.columns]
        else:
            base_datasets = Parallel(n_jobs)(delayed(
                self._create_dataset)(
                descriptions[i], target_name, preload, add_physician_reports
            ) for i in descriptions.columns)
        super().__init__(base_datasets)

    @staticmethod
    def _create_dataset(description, target_name, preload,
                        add_physician_reports):
        file_path = description.loc['path']

        # parse age and gender information from EDF header
        age, gender = _parse_age_and_gender_from_edf_header(file_path)
        raw = mne.io.read_raw_edf(file_path, preload=preload)

        # Use recording date from path as EDF header is sometimes wrong
        meas_date = datetime(1, 1, 1, tzinfo=timezone.utc) \
            if raw.info['meas_date'] is None else raw.info['meas_date']
        raw.set_meas_date(meas_date.replace(
            *description[['year', 'month', 'day']]))

        # read info relevant for preprocessing from raw without loading it
        d = {
            'age': int(age),
            'gender': gender,
        }
        if add_physician_reports:
            physician_report = _read_physician_report(file_path)
            d['report'] = physician_report
        additional_description = pd.Series(d)
        description = pd.concat([description, additional_description])
        base_dataset = BaseDataset(raw, description,
                                   target_name=target_name)
        return base_dataset

class TUHAbnormal(TUH):
    """Temple University Hospital (TUH) Abnormal EEG Corpus.
    see www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tuab
    Parameters
    ----------
    path: str
        Parent directory of the dataset.
    recording_ids: list(int) | int
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    target_name: str
        Can be 'pathological', 'gender', or 'age'.
    preload: bool
        If True, preload the data of the Raw objects.
    add_physician_reports: bool
        If True, the physician reports will be read from disk and added to the
        description.
    """
    def __init__(self, path, recording_ids=None, target_name='pathological',
                 preload=False, add_physician_reports=False, n_jobs=1):
        super().__init__(path=path, recording_ids=recording_ids,
                         preload=preload, target_name=target_name,
                         add_physician_reports=add_physician_reports,
                         n_jobs=n_jobs)
        additional_descriptions = []
        for file_path in self.description.path:
            additional_description = (
                self._parse_additional_description_from_file_path(file_path))
            additional_descriptions.append(additional_description)
        additional_descriptions = pd.DataFrame(additional_descriptions)
        self.set_description(additional_descriptions, overwrite=True)

    @staticmethod
    def _parse_additional_description_from_file_path(file_path):
        file_path = os.path.normpath(file_path)
        tokens = file_path.split(os.sep)
        # expect paths as version/file type/data_split/pathology status/
        #                     reference/subset/subject/recording session/file
        # e.g.            v2.0.0/edf/train/normal/01_tcp_ar/000/00000021/
        #                     s004_2013_08_15/00000021_s004_t000.edf
        assert ('abnormal' in tokens or 'normal' in tokens), (
            'No pathology labels found.')
        assert ('train' in tokens or 'eval' in tokens), (
            'No train or eval set information found.')
        return {
            'version': 'V3.0',
            'train': 'train' in tokens,
            'pathological': 'abnormal' in tokens,
        }

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
    if 'EEG' in ch_name:
        out = ch_name.replace('EEG ', '').replace('-REF', '')
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

    # Make up birthday based on recording date and age to allow mne-bids to
    # compute age
    birthday = datetime.datetime(desc['year'] - desc['age'], desc['month'], 1)
    birthday -= datetime.timedelta(weeks=4)
    sex = desc['gender'].lower()  # This assumes gender=sex

    # Add additional data required by BIDS
    mrn = str(desc['subject']).zfill(4)  # MRN: Medical Record Number
    session_nb = str(desc['session']).zfill(3)
    subject_info = {
        'participant_id': mrn,
        'subject': desc['subject_orig'],
        'birthday': (birthday.year, birthday.month, birthday.day),
        'sex': SEX_TO_MNE[sex],
        'train': desc['train'],
        'pathological': desc['pathological'],
        'handedness': None  # Not available
    }
    raw.info['line_freq'] = 60.  # Data was collected in North America
    raw.info['subject_info'] = subject_info
    task = 'rest'

    bids_path = BIDSPath(
        subject=mrn, session=session_nb, task=task, run=desc['segment'],
        root=bids_save_dir, datatype='eeg', check=True)

    write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True,
                   format='BrainVision')


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

    concat_ds = TUHAbnormal(tuh_data_dir, recording_ids=None, n_jobs=16)
    subjects = concat_ds.description.subject.astype('category').cat.codes
    concat_ds.set_description({'subject_orig': concat_ds.description.subject})
    concat_ds.set_description({'subject': subjects}, overwrite=True)
    
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
        _convert_tuh_recording_to_bids(
            ds, bids_save_dir, desc=desc)


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
