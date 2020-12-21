import json
from collections import OrderedDict
import os
from shutil import SameFileError

import mne
import mne_bids
import numpy as np
import pandas as pd


def get_subfolders(subject_path, verbose=True):
    """
    given an address, provide all the subfolders included in such path.
    this function is called during get_address_ieeg_files function

    Parameters
    ----------
    subject_path : string
        address to the subject folder
    verbose : boolean, optional

    Returns
    -------
    subfolders : list


    """
    subfolders = []
    for entry in os.listdir(subject_path):
        if os.path.isdir(os.path.join(subject_path, entry)):
            subfolders.append(entry)
            if verbose:
                print(entry)

    return subfolders


def get_address_ieeg_files(subject_path, subfolder, verbose=True):
    """Given an address to a subject folder and a list of subfolders, provide a list of all ieeg files
    recorded for that particular subject.

    To access to a particular vhdr_file please see 'read_bids_file'.

    To get info from vhdr_file please see 'get_subject_sess_task_run'


    Parameters
    ----------
    subject_path : string
    subfolder : list
    verbose : boolean, optional

    Returns
    -------
    vhdr_files : list
        list of addrress to access to a particular vhdr_file.


    """
    vhdr_files = []
    for i in range(len(subfolder)):
        session_path = subject_path + subfolder[i] + '/ieeg'
        for f_name in os.listdir(session_path):
            if f_name.endswith('.vhdr'):
                vhdr_files.append(session_path + '/' + f_name)
                if verbose:
                    print(f_name)
    return vhdr_files


def get_files(subject_path, subfolder, endswith=".vhdr", verbose=True):
    """Given an address to a subject folder and a list of subfolders, provide a list of all vhdr files
    recorded for that particular subject.

    To access to a particular vhdr_file please see 'read_bids_file'.

    To get info from vhdr_file please see 'get_subject_sess_task_run'


    Parameters
    ----------
    subject_path : string
    subfolder : list
    endswith : file extension (optional). Default = ".vhdr"
    verbose : boolean, optional

    Returns
    -------
    vhdr_files : list
        list of addrress to access to a particular vhdr_file.


    """
    vhdr_files = []
    for i in range(len(subfolder)):
        session_path = subject_path + '/' + subfolder[i] + '/ieeg'
        for f_name in os.listdir(session_path):
            if f_name.endswith(endswith):
                vhdr_files.append(session_path + '/' + f_name)
                if verbose:
                    print(f_name)
    return vhdr_files


def get_all_files(path, suffix, get_bids=False, prefix=None, bids_root=None):
    """Return all files in all (sub-)directories of path with given suffixes and prefixes (case-insensitive).

    Args:
        path (string)
        suffix (iterable): e.g. ["vhdr", "edf"] or [".json"]
        get_bids (boolean): True if BIDS_Path type should be returned instead of string. Default: False
        bids_root (string/path): Path of BIDS root folder. Only required if get_bids=True.
        prefix (iterable): e.g. ["SelfpacedRota", "ButtonPress] (optional)

    Returns:
        filepaths (list of strings or list of BIDS_Path)
    """
    filepaths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            for suff in suffix:
                if file.endswith(suff.lower()):
                    if prefix is not None:
                        for pref in list(prefix):
                            if pref.lower() in file.lower():
                                filepaths.append(os.path.join(root, file))
                    else:
                        filepaths.append(os.path.join(root, file))
    if not filepaths:
        print("No files found.")
    if get_bids:
        bids_paths = []
        for filepath in filepaths:
            subject, session, task, run = get_subject_sess_task_run(filepath)
            bids_path = mne_bids.BIDSPath(subject=subject, session=session, task=task, run=run, datatype="ieeg",
                                          root=bids_root)
            bids_paths.append(bids_path)
        return bids_paths
    else:
        return filepaths


def get_all_bids_paths(path, suffix, prefix=None):
    """Return all files in all (sub-)directories of path with given suffixes and prefixes (case-insensitive).

    Args:
        path (string)
        suffix (iterable): e.g. ["vhdr", "edf"] or [".json"]
        prefix (iterable): e.g. ["SelfpacedRota", "ButtonPress] (optional)
    Returns:
        filepaths (list of strings)
    """
    filepaths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            for suff in suffix:
                if file.endswith(suff.lower()):
                    if prefix is not None:
                        for pref in list(prefix):
                            if pref.lower() in file.lower():
                                filepaths.append(os.path.join(root, file))
                    else:
                        filepaths.append(os.path.join(root, file))
    if not filepaths:
        print("No files found.")
    return filepaths


def read_bids_file(file_path):
    """Read one run file from BIDS standard

    :param file_path: .vhdr file
    :return: raw dataset array, channel name array
    """
    bv_file = mne_bids.read.io.brainvision.read_raw_brainvision(file_path)
    bv_raw = bv_file.get_data()
    return bv_raw, bv_file.ch_names


def read_M1_channel_specs(run_string):
    """Given a run file, read the respective M1 channel specification file in format ch_name, rereference, used, \
    predictor and return a dict with "cortex", "subcortex" and "labels" channels. If no cortex/subcortex channels \
    are used, they are None.

    Args:
        run_string (string): run string without specific ending in form sub-000_ses-right_task-force_run-0
    Returns:
        used_channels
    """

    df_channel = pd.read_csv(run_string + "_channels_M1.tsv", sep="\t")

    # used channels is a dict though with 'cortex' and 'subcortex' field
    ch_cortex = np.array([ch_idx for ch_idx, ch in enumerate(df_channel['name']) if 'ECOG' in ch
                          and ch_idx in np.where(df_channel['used'] == 1)[0]])
    ch_subcortex = np.array([ch_idx for ch_idx, ch in enumerate(df_channel['name']) if 'STN' in ch
                             and ch_idx in np.where(df_channel['used'] == 1)[0]])  # needs to be specified

    used_channels = {
        "cortex": None,
        "subcortex": None,
        "labels": None
    }

    if ch_cortex.shape[0] != 0:
        used_channels["cortex"] = ch_cortex

    if ch_subcortex.shape[0] != 0:
        used_channels["subcortex"] = ch_subcortex

    ch_labels = np.where(df_channel["target"] == 1)[0]

    if ch_labels.shape[0] != 0:
        used_channels["labels"] = ch_labels

    return used_channels


def read_grid():
    """
    Args:
    Returns:
    """

    cortex_left = np.array(pd.read_csv('settings/cortex_left.tsv', sep="\t"))
    cortex_right = np.array(pd.read_csv('settings/cortex_right.tsv', sep="\t"))
    subcortex_left = np.array(pd.read_csv('settings/subcortex_left.tsv', sep="\t"))
    subcortex_right = np.array(pd.read_csv('settings/subcortex_right.tsv', sep="\t"))
    return cortex_left.T, cortex_right.T, subcortex_left.T, subcortex_right.T


def get_coords_df_from_vhdr(vhdr_file, bids_path):
    """Given a vhdr file path and the BIDS path return a pandas dataframe of that session (important: not for the run; \
    run channels might have only a subset of all channels in the coordinate file)

    Args:
        vhdr_file (string)
        bids_path
    Returns:
        df (pandas dataframe)
    """

    subject = vhdr_file[vhdr_file.find('sub-') + 4:vhdr_file.find('sub-') + 7]

    if vhdr_file.find('right') != -1:
        sess = 'right'
    else:
        sess = 'left'
    coord_path = os.path.join(bids_path, 'sub-' + subject, 'ses-' + sess, 'ieeg', 'sub-' + subject + '_electrodes.tsv')
    df = pd.read_csv(coord_path, sep="\t")
    return df


def read_run_sampling_frequency(vhdr_file):
    """Given a .eeg vhdr file, read the respective channel file and return the the sampling frequency for the first
    index, since all channels are throughout the run recorded with the same sampling frequency.
    """

    ch_file = vhdr_file[:-9] + 'channels.tsv'
    df = pd.read_csv(ch_file, sep="\t")
    return df['sampling_frequency']


def read_line_noise(bids_path, subject):
    """Return the line noise for a given subject (in shape '000') from participants.tsv.
    """
    df = pd.read_csv(os.path.join(bids_path, 'participants.tsv'), sep="\t")
    row_ = np.where(df['participant_id'] == 'sub-' + str(subject))[0][0]
    return df.iloc[row_]['line_noise']


def get_patient_coordinates(ch_names, ind_cortex, ind_subcortex, vhdr_file, bids_path):
    """For a given vhdr file, the respective BIDS path, and the used channel names of a BIDS run.

    :return the coordinate file of the used channels
        in shape (2): cortex; subcortex; fields might be empty (None if no cortex/subcortex channels are existent)
        appart from that the used fields are in numpy array field shape (num_coords, 3)
    """

    df = get_coords_df_from_vhdr(vhdr_file, bids_path)  # this dataframe contains all coordinates in this session
    coord_patient = np.empty(2, dtype=object)

    if ind_cortex is not None:
        coord_cortex = np.zeros([ind_cortex.shape[0], 3])
        for idx, ch_cortex in enumerate(np.array(ch_names)[ind_cortex]):
            coord_cortex[idx, :] = np.array(df[df['name'] == ch_cortex].iloc[0][1:4]).astype('float')
        coord_patient[0] = coord_cortex

    if ind_subcortex is not None:
        coord_subcortex = np.zeros([ind_subcortex.shape[0], 3])
        for idx, ch_subcortex in enumerate(np.array(ch_names)[ind_subcortex]):
            coord_subcortex[idx, :] = np.array(df[df['name'] == ch_subcortex].iloc[0][1:4]).astype('float')
        coord_patient[1] = coord_subcortex

    return coord_patient


def get_active_grid_points(sess_right, ind_label, ch_names, proj_matrix_run, grid_):
    """
    params:
        sess_right: boolean that determines if the session is left or right
        ch_names : list from brainvision
        proj_matrix_run : list: 0 - cortex; 1 - subcortex, projection array in shape grid_points X channels;

    returns:
        arr_act_grid_points: array in shape num grids points cortex_LEFT + subcortex_LEFT + cortex_RIGHT \
        + subcortex_RIGHT. 0/1 indication for used interpolation or not
    """

    arr_act_grid_points = np.zeros([grid_[0].shape[1] + grid_[1].shape[1] + grid_[2].shape[1] + grid_[3].shape[1]])
    label_channel = np.array(ch_names)[ind_label]
    con_label = False
    ips_label = False

    # WRITE CONTRALATERAL DATA if the respective movement channel exists
    if sess_right is True:
        if len([ch for ch in label_channel if 'LEFT' in ch]) > 0:
            con_label = True
    elif sess_right is False:
        if len([ch for ch in label_channel if 'RIGHT' in ch]) > 0:
            con_label = True

    # WRITE IPSILATERAL DATA if the respective movement channel exists
    if sess_right is False:
        if len([ch for ch in label_channel if 'LEFT' in ch]) > 0:
            ips_label = True
    elif sess_right is True:
        if len([ch for ch in label_channel if 'RIGHT' in ch]) > 0:
            ips_label = True

    if con_label is True and proj_matrix_run[0] is not None:
        arr_act_grid_points[np.nonzero(np.sum(proj_matrix_run[0], axis=1))[0]] = 1
    if ips_label is True and proj_matrix_run[0] is not None:
        arr_act_grid_points[np.nonzero(np.sum(proj_matrix_run[0], axis=1))[0] + grid_[0].shape[1]] = 1
    if con_label is True and proj_matrix_run[1] is not None:
        arr_act_grid_points[np.nonzero(np.sum(proj_matrix_run[1], axis=1))[0] + grid_[0].shape[1] * 2] = 1
    if ips_label is True and proj_matrix_run[1] is not None:
        arr_act_grid_points[np.nonzero(np.sum(proj_matrix_run[1], axis=1))[0] + grid_[0].shape[1] * 2
                            + grid_[1].shape[1]] = 1

    return arr_act_grid_points


def get_subject_sess_task_run(vhdr_file):
    """ Given a BIDS-conform filename return the corresponding subject, session, task and run.

    Args:
        vhdr_file (string): Name of file
    Return:
        subject, sess, task, run (strings)
    """

    subject = vhdr_file[vhdr_file.rfind('sub-') + 4:vhdr_file.rfind('ses') - 1]

    str_sess = vhdr_file[vhdr_file.rfind('ses'):]
    sess = str_sess[str_sess.find('-') + 1:str_sess.find('_')]

    str_task = vhdr_file[vhdr_file.rfind('task'):]
    task = str_task[str_task.find('-') + 1:str_task.find('run') - 1]

    str_run = vhdr_file[vhdr_file.rfind('run'):]
    run = min(str_run[str_run.find('-') + 1:str_run.find('_')], str_run[str_run.find('-') + 1:str_run.find('.')])

    return subject, sess, task, run


def get_used_ch_idx(used_channels, ch_names_bv):
    """read from the provided list of used_channels the indices of channels in the channel names brainvision file

    Args:
        used_channels (list): used channels for a given location/type, e.g. cortical, subcortical, label
        ch_names_bv (list): channel names of brainvision file

    Returns:
        np array: indizes of channel names in ch_names_BV
    """
    used_idx = []
    for ch_track in used_channels:
        for ch_idx, ch in enumerate(ch_names_bv):
            if ch == ch_track:
                used_idx.append(ch_idx)
    return np.array(used_idx)


def get_dat_cortex_subcortex(bv_raw, used_channels):
    """
    Data segmentation into cortex, subcortex, MOV and dat; returns also respective indizes of bv_raw
    :param bv_raw: raw np.array of Brainvision-read file
    :param used_channels
    """

    data_ = {
        "dat_cortex": None,
        "dat_subcortex": None,
        "dat_label": None,
        "ind_cortex": used_channels['cortex'],
        "ind_subcortex": used_channels['subcortex'],
        "ind_label": used_channels['labels'],
        "ind_dat": None
    }

    if used_channels['cortex'] is not None:
        data_["dat_cortex"] = bv_raw[data_["ind_cortex"], :]

    if used_channels['subcortex'] is not None:
        data_["dat_subcortex"] = bv_raw[data_["ind_subcortex"], :]

    if used_channels['labels'] is not None:
        data_["dat_label"] = bv_raw[data_["ind_label"], :]
        data_["ind_dat"] = np.arange(bv_raw.shape[0])[~np.isin(np.arange(bv_raw.shape[0]), data_["ind_label"])]

    return data_


def read_settings(file_name=None):
    if file_name is None:
        file_name = 'settings'

    with open('settings/' + file_name + '.json', 'rb') as f:
        settings = json.load(f)
    return settings


def sess_right(sess):
    if 'right' in sess:
        sess_right = True
    else:
        sess_right = False
    return sess_right


def write_all_M1_channel_files(settings, cortex_ref='average', subcortex_ref='-'):
    """

    Read all channels.tsv in the settings defined BIDS path, and write all all channels_M1.tsv files
    --> copy all channel names from channel.tsv as name
    --> set targets to 'MOV' channels
    --> rereference all cortex to average, subcortex=None
    --> used all to 1

    """

    # settings = IO.read_settings()  # reads settings from settings/settings.json file in a dict
    BIDS_channel_tsv_files = []
    for root, dirs, files in os.walk(settings['bids_path']):
        for file in files:
            if file.endswith("_channels.tsv"):
                ch_file = os.path.join(root, file)
                df_channel = pd.read_csv(ch_file, sep="\t")
                df = pd.DataFrame(np.nan, index=np.arange(len(list(df_channel['name']))),
                                  columns=['name', 'rereference', 'used', 'target'])
                df['used'] = 1
                df['name'] = list(df_channel['name'].copy(deep=True))

                ch_mov = [ch_idx for ch_idx, ch in enumerate(df_channel['name']) if ch.startswith('MOV')]
                target = np.zeros(len(list(df_channel['name'])))
                target[ch_mov] = 1
                df['target'] = target.astype(int)

                rereference = ["" for x in range(len(list(df_channel['name'])))]

                for ch_idx, ch in enumerate(df_channel['name']):
                    if ch.startswith('ECOG'):
                        rereference[ch_idx] = cortex_ref
                    if ch.startswith('STN'):
                        rereference[ch_idx] = subcortex_ref

                df['rereference'] = rereference
                df.to_csv(ch_file[:-12] + 'channels_M1.tsv', sep='\t')
                BIDS_channel_tsv_files.append(ch_file)
    return BIDS_channel_tsv_files


def write_bids(bids_root, filename, outpath, set_chtypes=True):
    """Write or overwrite existing BIDS files from raw brainvision file, organized in BIDS structure.

    Keyword arguments
    -----------------
    bids_root (string): Path to folder of BIDS root (e.g. '/Users/johndoe/BIDS/')
    filename (string): Name of file with BIDS-compliant filename (e.g. 'sub-1_ses-1_task-rest_run_1')
    outpath (string): Path to folder of output BIDS files (e.g. '/Users/johndoe/BIDS_2')
    set_chtypes (boolean): (Optional) If set to True, reset channel types (default: True)
    electr_file (string): (Optional) Path to file containing information about electrode localizations (default: None)

    Returns
    -------
    None
    """

    subject, session, task, run = get_subject_sess_task_run(filename)
    dataype = 'ieeg'
    bids_in = mne_bids.BIDSPath(subject=subject, session=session, task=task, run=run, datatype=dataype, root=bids_root)
    bids_out = mne_bids.BIDSPath(subject=subject, session=session, task=task, run=run, datatype=dataype, root=outpath)

    # If preload is set to TRUE, write_raw_bids might not work. Only load_data if necessary.
    try:
        raw = mne_bids.read_raw_bids(bids_path=bids_in, extra_params=dict(preload=False), verbose=False)
    except:
        print('Possible error in BIDS structure, try mne.io.read_raw_brainvision and .read_raw_edf to read in file...')
        file = str(bids_in.fpath)
        if file.endswith(".vhdr"):
            raw = mne_bids.read.io.brainvision.read_raw_brainvision(file, preload=False, verbose=False)
        elif file.endswith(".edf"):
            raw = mne.io.read_raw_edf(file, preload=False, verbose=False)
        else:
            print('File could not be treated: ', bids_in.basename)
            return
        for f_name in os.listdir(bids_in.directory):
            if f_name.endswith('ieeg.json'):
                bids_json = bids_in.directory / f_name
        with open(bids_json, 'rb') as f:
            settings = json.load(f)
        raw.info['line_freq'] = settings['PowerLineFrequency']

    if set_chtypes:
        print('Setting new channel types...')
        remapping_dict = {}
        for ch_name in raw.info['ch_names']:
            if ch_name.startswith('ECOG'):
                remapping_dict[ch_name] = 'ecog'
            elif ch_name.startswith('LFP') or ch_name.startswith('STN'):
                remapping_dict[ch_name] = 'seeg'
            elif ch_name.startswith('EMG'):
                remapping_dict[ch_name] = 'emg'
            # mne_bids cannot handle both eeg and ieeg channel types in the same data
            elif ch_name.startswith('EEG'):
                remapping_dict[ch_name] = 'misc'
            elif ch_name.startswith('MOV') or ch_name.startswith('ANALOG') or ch_name.startswith('ROT') \
                    or ch_name.startswith('ACC') or ch_name.startswith('AUX') or ch_name.startswith('X') \
                    or ch_name.startswith('Y') or ch_name.startswith('Z'):
                remapping_dict[ch_name] = 'misc'
        raw.set_channel_types(remapping_dict, verbose=False)

    electr_file = None
    for f_name in os.listdir(bids_in.directory):
        if f_name.endswith('sub-' + subject + '_electrodes.tsv') \
                or f_name.endswith('ses-' + session + '_electrodes.tsv'):
            electr_file = bids_in.directory / f_name

    if electr_file is not None:
        print('Electrodes file being used: ', electr_file.name)
        data = np.loadtxt(str(electr_file), dtype=str, delimiter='\t', comments=None, encoding='utf-8')
        column_names = data[0, :]
        info = data[1:, :]

        electrode_tsv = OrderedDict()
        for i, name in enumerate(column_names):
            electrode_tsv[name] = info[:, i].tolist()

        # Load in channel names
        ch_names = electrode_tsv['name']

        # Load in the xyz coordinates as a float
        elec = np.empty(shape=(len(ch_names), 3))
        try:
            for ind, axis in enumerate(['x', 'y', 'z']):
                elec[:, ind] = list(map(float, electrode_tsv[axis]))
        except:
            try:
                for ind, axis in enumerate(['x_MNI', 'y_MNI', 'z_MNI']):
                    elec[:, ind] = list(map(float, electrode_tsv[axis]))
            except:
                print('No electrode coordinates found.')
        elec = elec / 1000  # convert mm to mne meter standard
    try:
        # Create mne montage
        montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)), coord_frame='mni_tal')
        # Set montage. Warning is issued if channels don't match. Consider getting locations for missing channels.
        raw.set_montage(montage, on_missing='warn', verbose=False)
    except:
        print('Montage was not possible.')

    # Write out files in BIDS format
    # Might issue SameFileError if no changes are made to raw. Can be ignored, since _ieeg files don't need to be \
    # overwritten.

    # Workaround, if raw data has been loaded.
    fname_fif = bids_in.directory / (bids_in.basename + 'raw.fif')
    if raw.preload:
        raw.save(fname_fif, proj=True, overwrite=True)
        raw = mne.io.read_raw_fif(fname_fif, preload=False, verbose=False)
    try:
        mne_bids.write_raw_bids(raw, bids_out, overwrite=True, verbose=False)
    except SameFileError:
        print('SameFileError was ignored.')
        pass

    # Erase file, if workaround was used
    if os.path.exists(fname_fif):
        os.remove(fname_fif)

    return
