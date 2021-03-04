from datetime import datetime
import os
import pathlib
import shutil

import numpy as np
import pandas as pd

import mne
from mne_bids import read_raw_bids
from pybv import write_brainvision

import icn_tb as tb


def bids_rewrite_file(raw, bids_path, return_raw=False):
    """Overwrite BrainVision data in BIDS format that has been modified.

    Parameters
    ----------
    raw : raw MNE object
        The raw MNE object for this function to write
    bids_path : BIDSPath MNE-BIDS object
        The MNE BIDSPath to the file to be overwritten
    return_raw : boolean, optional
        Set to True to return the new raw object that has been written.
        Default is False.
    Returns
    -------
    raw_new : raw MNE object or None
        The newly written raw object.
    """
    # this is necessary to ensure to write the correct BrainVision files
    bids_path.update(suffix=bids_path.datatype)

    data = raw.get_data()
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    fname_base = bids_path.basename
    folder_out = os.path.join(bids_path.directory, 'dummy')
    events, event_id = mne.events_from_annotations(raw)
    events_new = np.vstack((events[:, 0], events[:, 2])).T

    write_brainvision(data=data, sfreq=sfreq, ch_names=ch_names,
                      fname_base=fname_base, folder_out=folder_out,
                      events=events_new)
    # now delete and move files
    suffixes = ['.eeg', '.vhdr', '.vmrk']
    orig_files = [os.path.join(bids_path.directory, bids_path.basename + suffix)
                  for suffix in suffixes]
    for orig_file in orig_files:
        os.remove(orig_file)
    new_files = [
        os.path.join(bids_path.directory, 'dummy', bids_path.basename + suffix)
        for suffix in suffixes]
    for new_file, orig_file in zip(new_files, orig_files):
        os.rename(new_file, orig_file)
    os.rmdir(os.path.join(bids_path.directory, 'dummy'))

    if return_raw is True:
        raw = read_raw_bids(bids_path)
        return raw


def bids_get_participants_tsv_filename(bids_folder):
    return str(pathlib.Path(bids_folder, 'participants.tsv'))


def bids_read_participants_tsv(bids_folder):
    return pd.read_csv(bids_get_participants_tsv_filename(bids_folder),
                       delimiter='\t')


def bids_backup_participants_tsv(bids_folder):
    fdir, fname, ext = \
        tb.fileparts(bids_get_participants_tsv_filename(bids_folder))
    shutil.copyfile(str(pathlib.Path(fdir, fname + ext)),
                    str(pathlib.Path(fdir, fname + '_backup' + ext)))


def bids_write_participants_tsv(df, bids_folder):
    bids_backup_participants_tsv(bids_folder)
    df.to_csv(bids_get_participants_tsv_filename(bids_folder), sep='\t',
              index=False)


def bids_get_participant_id_from_filename(filename):
    fdir, fname, ext = tb.fileparts(filename)
    dirparts = pathlib.Path(fdir).parts
    for item in list(dirparts):
        if item.startswith('sub-'):
            return item


def bids_get_participants(bids_folder, item='participant_id'):
    df = bids_read_participants_tsv(bids_folder)
    return list(df[item])


def bids_write_json_to_participants_tsv(bids_folder, json_file,
                                        participant_id=None):
    if participant_id is None:
        participant_id = bids_get_participant_id_from_filename(json_file)
    js = pd.read_json(json_file, typ='series')
    subs = bids_read_participants_tsv(bids_folder)
    i = [i for i, x in enumerate(subs.participant_id == participant_id) if x]
    for k in js.keys():
        print(k)
        print(str(js[k]))
        if js[k]:
            subs.loc[i, k] = str(js[k])
    subs.to_csv(bids_get_participants_tsv_filename(bids_folder), sep='\t',
                index=False)
    fdir, fname, ext = tb.fileparts(str(pathlib.Path(bids_folder,
                                                     'participants.json')))
    shutil.copyfile(pathlib.Path(fdir, fname + ext),
                    pathlib.Path(fdir, fname + '_backup' + ext))
    js = tb.json_read(pathlib.Path(fdir, fname + ext))
    for a in subs.keys():
        if a not in list(js.keys()):
            js.update({a: {'Description': a}})
    tb.json_write(pathlib.Path(fdir, fname + ext), js)
    return subs


def date_string():
    return f'{datetime.now():%Y_%m_%d_%H_%M_%S%z}'
