import argparse
import os
import pathlib
from tkinter import BOTTOM
import urllib.request
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import mne
from mne.io.brainvision.brainvision import _aux_vhdr_info

from mne_bids import write_raw_bids, print_dir_tree, make_report, BIDSPath

lemon_info = pd.read_csv(
    "./META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv")
lemon_info = lemon_info.set_index("ID")
eeg_subjects = pd.read_csv('./lemon_eeg_subjects.csv')
lemon_info = lemon_info.loc[eeg_subjects.subject]
lemon_info['gender'] = lemon_info['Gender_ 1=female_2=male'].map({1: 2, 2: 1})
subjects = list(lemon_info.index)

def convert_lemon_to_bids(lemon_data_dir, bids_save_dir, n_jobs=1, DEBUG=False):
    """Convert TUAB dataset to BIDS format.

    Parameters
    ----------
    lemon_data_dir : str
        Directory where the original LEMON dataset is saved, e.g.
        `/storage/store3/data/LEMON_RAW`.
    bids_save_dir : str
        Directory where to save the BIDS version of the dataset.
    n_jobs : None | int
        Number of jobs for parallelization.
    """
    subjects_ = subjects
    if DEBUG:
        subjects_ = subjects[:1]

    good_subjects = Parallel(n_jobs=n_jobs)(
        delayed(_convert_subject)(subject, lemon_data_dir, bids_save_dir)
        for subject in subjects_) 
    subjects_ = [sub for sub in good_subjects if not isinstance(sub, tuple)]
    _, bad_subjects, errs = zip(*[
        sub for sub in good_subjects if isinstance(sub, tuple)])
    bad_subjects = pd.DataFrame(
        dict(subjects= bad_subjects, error=errs))
    bad_subjects.to_csv(
        '/storage/store3/data/LEMON_EEG_BIDS/bids_conv_erros.csv')
    # update the participants file as LEMON has no official age data
    participants = pd.read_csv(
        "/storage/store3/data/LEMON_EEG_BIDS/participants.tsv", sep='\t')
    participants = participants.set_index("participant_id")
    participants.loc[subjects_, 'age'] = lemon_info.loc[subjects_, 'age']
    participants.to_csv(
        "/storage/store3/data/LEMON_EEG_BIDS/participants.tsv", sep='\t')


def _convert_subject(subject, data_path, bids_save_dir):
    """Get the work done for one subject"""
    try:
        fname = pathlib.Path(data_path) / subject / "RSEEG" / f"{subject}.vhdr"    
        raw = mne.io.read_raw_brainvision(fname)

        raw.set_channel_types({"VEOG": "eog"})
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage)
        sub_id = subject.strip("sub-")
        raw.info['subject_info'] = {
            'participant_id': sub_id,
            'sex': lemon_info.loc[subject, 'gender'],
            'age': lemon_info.loc[subject, 'age'],
            # XXX LEMON shares no public age 
            'hand': lemon_info.loc[subject, 'Handedness']
        }
        events, _ = mne.events_from_annotations(raw)

        events = events[(events[:, 2] == 200) | (events[:, 2] == 210)]
        event_id = {"eyes/open": 200, "eyes/closed": 210}
        bids_path = BIDSPath(
            subject=sub_id, session=None, task='RSEEG',
            run=None,
            root=bids_save_dir, datatype='eeg', check=True)

        write_raw_bids(
            raw,
            bids_path,
            events_data=events,
            event_id=event_id,
            overwrite=True
        )
    except Exception as err:
        print(err)
        return ("BAD", subject, err)
    return subject


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert LEMON to BIDS.')
    parser.add_argument(
        '--lemon_data_dir', type=str,
        default='/storage/store3/data/LEMON_RAW',
        help='Path to the original data.')
    parser.add_argument(
        '--bids_data_dir', type=str,
        default=pathlib.Path("/storage/store3/data/LEMON_EEG_BIDS"),
        help='Path to where the converted data should be saved.')
    parser.add_argument(
        '--n_jobs', type=int, default=1,
        help='number of parallel processes to use (default: 1)')
    parser.add_argument(
        '--DEBUG', type=bool, default=False,
        help='activate debugging mode')
    args = parser.parse_args()

    age_info = pd.read_csv(args.bids_data_dir + '/participants.tsv',
                           sep='\t')
    lemon_info['age'] = age_info['Age']
    convert_lemon_to_bids(
        args.lemon_data_dir, args.bids_data_dir, n_jobs=args.n_jobs,
        DEBUG=args.DEBUG)

    print_dir_tree(args.bids_data_dir)
    print(make_report(args.bids_data_dir))
