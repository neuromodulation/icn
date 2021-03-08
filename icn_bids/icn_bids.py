from datetime import datetime
import os
import pathlib
import shutil

import numpy as np
import pandas as pd

import mne
from mne_bids import read_raw_bids
from mne_bids.copyfiles import copyfile_brainvision
from bids import BIDSLayout
from pybv import write_brainvision

import icn_tb as tb


def get_electrodes(sub, ch_type, space, bids_path):
    """Returns pandas dataframe of electrodes of a given subject in a BIDS root directory

    Args:
        sub (string): BIDS subject
        ch_type (string): BIDS specific channel type in channels.tsv
        space (string) : BIDS specific electrode space
        bids_path (string): BIDS root directory

    Returns:
        pandas dataframe: run concatenated electrode tsv dataframe for given subject and channel type
    """
    layout = BIDSLayout(bids_path)

    channels = layout.get(subject=sub, extension='.tsv', suffix='channels')
    electrodes = layout.get(subject=sub, space=space, extension='.tsv', suffix='electrodes')

    if len(channels) != len(electrodes):
        assert False, "channel.tsv length and electrodes.tsv does not match"

    df_electrodes = pd.DataFrame()

    for channel_file, electrode_file in zip(channels, electrodes):
        df_run = electrode_file.get_df()[np.array(channel_file.get_df()["type"] == ch_type)]
        df_electrodes = pd.concat([df_electrodes,df_run]).drop_duplicates().reset_index(drop=True)    
        
    return df_electrodes


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
    bids_path.update(suffix=bids_path.datatype)

    data = raw.get_data()
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    fname = bids_path.basename
    folder = bids_path.directory
    events, event_id = mne.events_from_annotations(raw)
    events_new = np.vstack((events[:, 0], events[:, 2])).T

    # rewrite datafile
    write_brainvision(data=data, sfreq=sfreq, ch_names=ch_names,
                      fname_base='dummy', folder_out=folder,
                      events=events_new)
    suffixes = ['.eeg', '.vhdr', '.vmrk']
    orig_files = [os.path.join(folder, fname + suffix)
                  for suffix in suffixes]
    for orig_file in orig_files:
        os.remove(orig_file)
    source_path = os.path.join(folder, 'dummy' + '.vhdr')
    dest_path = os.path.join(folder, fname + '.vhdr')
    copyfile_brainvision(source_path, dest_path)
    dummy_files = [os.path.join(folder, 'dummy' + suffix)
                   for suffix in suffixes]
    for dummy_file in dummy_files:
        os.remove(dummy_file)

    # rewrite events.tsv
    channels_path = bids_path.copy().update(suffix='channels')
    channels_tsv = channels_path.fpath
    df = pd.read_csv(channels_tsv, sep='\t', index_col=0)
    old_chs = df.index.tolist()
    add_chs = [ch for ch in raw.ch_names if ch not in old_chs]
    description = {'seeg': 'StereoEEG', 'ecog': 'Electrocorticography',
                   'eeg': 'Electroencephalography', 'emg': 'Electromyography',
                   'misc': 'Miscellaneous', 'dbs': 'Deep Brain Stimulation'}
    add_list = []
    ch_types = raw.get_channel_types(picks=add_chs)
    print(add_chs)
    print(ch_types)
    for idx, add_ch in enumerate(add_chs):
        add_dict = {}
        add_dict.update({df.columns[i]: df.iloc[0][i]
                         for i in range(0, len(df.columns))})
        add_dict.update({'type': ch_types[idx].upper()})
        add_dict.update({'description': description.get(ch_types[idx])})
        add_list.append(add_dict)
    index = pd.Index(add_chs, name='name')
    df_add = pd.DataFrame(add_list, index=index)
    df = df.append(df_add, ignore_index=False)
    remov_chs = [ch for ch in old_chs if ch not in raw.ch_names]
    df = df.drop(remov_chs)
    df = df.reindex(raw.ch_names)
    os.remove(channels_tsv)
    df.to_csv(os.path.join(folder, channels_path.basename + '.tsv'),
              sep='\t', index=True)
    # rewrite **electrodes.tsv
    elec_files = []
    for file in os.listdir(folder):
        if file.endswith('_electrodes.tsv') and '_space-' in file:
            elec_files.append(os.path.join(folder, file))
    for elec_file in elec_files:
        df = pd.read_csv(elec_file, sep='\t', index_col=0)
        old_chs = df.index.tolist()
        add_chs = [ch for ch in raw.ch_names if ch not in old_chs]
        add_list = []
        for add_ch in add_chs:
            add_dict = {}
            add_dict.update({column: 'n/a' for column in df.columns})
            add_list.append(add_dict)
        index = pd.Index(add_chs, name='name')
        df_add = pd.DataFrame(add_list, index=index)
        df = df.append(df_add, ignore_index=False)
        remov_chs = [ch for ch in old_chs if ch not in raw.ch_names]
        df = df.drop(remov_chs)
        df = df.reindex(raw.ch_names)
        os.remove(elec_file)
        df.to_csv(os.path.join(elec_file), sep='\t', na_rep='n/a',
                  index=True)
    # check for success
    raw = read_raw_bids(bids_path, verbose=False)
    if return_raw is True:
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
