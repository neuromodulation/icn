import mne_bids
import numpy as np
import os
import pandas as pd
import json
import IO


def get_subfolders(subject_path, Verbose=True):
    """
    given and address, provides all the subfolder included in such path.
    this function is called during get_address_vhdr_files function    

    Parameters
    ----------
    subject_path : string
        address to the subject folder
    Verbose : boolean, optional

    Returns 
    -------
    subfolders : list
    

    """
    subfolders=[]
    for entry in os.listdir(subject_path):
        if os.path.isdir(os.path.join(subject_path, entry)):
            subfolders.append(entry)
            if Verbose: print(entry)
            
    return subfolders

def get_address_vhdr_files(subject_path, subfolder, Verbose=True):
    """
    given an address to a subject folder and a list of subfolders, provides a list of all vhdr files
    recorded for that particular subject.
    
    To access to a particular vhdr_file please see 'read_BIDS_file'.
    
    To get info from vhdr_file please see 'get_sess_run_subject'

    
    Parameters
    ----------
    subject_path : string
    subfolder : list
    Verbose : boolean, optional

    Returns
    -------
    vhdr_files : list
        list of addrress to access to a particular vhdr_file.
        

    """
    vhdr_files=[]
    for i in range(len(subfolder)):
        session_path=subject_path+subfolder[i]+'/ieeg'
        for f_name in os.listdir(session_path):
            if f_name.endswith('.vhdr'):
                vhdr_files.append(session_path+ '/' +f_name)
                if Verbose: print(f_name)
    return vhdr_files

def get_files(subject_path, subfolder, endswith='.vhdr', Verbose=True):
    """
    given an address to a subject folder and a list of subfolders, provides a list of all vhdr files
    recorded for that particular subject.
    
    To access to a particular vhdr_file please see 'read_BIDS_file'.
    
    To get info from vhdr_file please see 'get_sess_run_subject'

    
    Parameters
    ----------
    subject_path : string
    subfolder : list
    Verbose : boolean, optional

    Returns
    -------
    vhdr_files : list
        list of addrress to access to a particular vhdr_file.
        

    """
    vhdr_files=[]
    for i in range(len(subfolder)):
        session_path=subject_path+'/'+subfolder[i]+'/ieeg'
        for f_name in os.listdir(session_path):
            if f_name.endswith(endswith):
                vhdr_files.append(session_path+ '/' +f_name)
                if Verbose: print(f_name)
    return vhdr_files

def get_all_vhdr_files(BIDS_path):
    """
    
    Given a BIDS path return all vhdr file paths without BIDS_Layout

    Args:
        BIDS_path (string)
    Returns: 
        vhdr_files (list)
    """
    vhdr_files = []
    for root, dirs, files in os.walk(BIDS_path):
        for file in files:
            if file.endswith(".vhdr"):
                vhdr_files.append(os.path.join(root, file))
    return vhdr_files
    

def read_BIDS_file(file_path):
    """
    Read one run file from BIDS standard
    :param file_path: .vhdr file
    :return: raw dataset array, channel name array
    """
    bv_file = mne_bids.read.io.brainvision.read_raw_brainvision(file_path)
    bv_raw = bv_file.get_data()
    return bv_raw, bv_file.ch_names

def read_M1_channel_specs(run_string):
    # given a run in from, sub-000_ses-right_task-force_run-0, the M1 channel specs file is in form sub-000_ses-right_task-force_run-0_channels_M1.tsv 
    """ 

    given a run file, read the respective M1 channel specification file in format ch_name, rereference, used, predictor and return a dict with "cortex", "subcortex" and "labels" channels 
    if no cortex/subcortex channels are used, they are None
    
    Args:
        run_string (string): run string without specific ending in form sub-000_ses-right_task-force_run-0
    returns: 
        used_channels
    """

    df_channel = pd.read_csv(run_string + "_channels_M1.tsv", sep="\t")
    

    # used channels is here a dict though with 'cortex' and 'subcortex' field
    ch_cortex = np.array([ch_idx for ch_idx, ch in enumerate(df_channel['name']) if 'ECOG' in ch and ch_idx in np.where(df_channel['used']==1)[0]])
    ch_subcortex = np.array([ch_idx for ch_idx, ch in enumerate(df_channel['name']) if 'STN' in ch and ch_idx in np.where(df_channel['used']==1)[0]])   # needs to be specified

    used_channels = {
        "cortex" : None,
        "subcortex" : None, 
        "labels" : None
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
    cortex_left = np.array(pd.read_csv('settings/cortex_left.tsv', sep="\t"))
    cortex_right = np.array(pd.read_csv('settings/cortex_right.tsv', sep="\t"))
    subcortex_left = np.array(pd.read_csv('settings/subcortex_left.tsv', sep="\t"))
    subcortex_right = np.array(pd.read_csv('settings/subcortex_right.tsv', sep="\t"))
    return cortex_left.T, cortex_right.T, subcortex_left.T, subcortex_right.T

def get_coords_df_from_vhdr(vhdr_file, BIDS_path):
    """
    given a vhdr file path and the BIDS path
    :return a pandas dataframe of that session (important: not for the run; run channls might have only a s
    subset of all channels in the coordinate file)
    """
    subject = vhdr_file[vhdr_file.find('sub-')+4:vhdr_file.find('sub-')+7]

    if vhdr_file.find('right') !=-1:
        sess = 'right'
    else:
        sess = 'left'
    coord_path = os.path.join(BIDS_path, 'sub-'+ subject, 'ses-'+ sess, 'ieeg', 'sub-'+ subject+ '_electrodes.tsv')
    df = pd.read_csv(coord_path, sep="\t")
    return df

def read_run_sampling_frequency(vhdr_file):
    """
    given a .eeg vhdr file, read the respective channel file and return the the sampling frequency for the first
    index, since all channels are throughout the run recorded with the same sampling frequency 
    """
    ch_file = vhdr_file[:-9]+'channels.tsv' # read out the channel
    df = pd.read_csv(ch_file, sep="\t")
    return df['sampling_frequency']  

def read_line_noise(BIDS_path, subject):
    """
    return the line noise for a given subject (in shape '000') from participants.tsv
    """
    df = pd.read_csv(BIDS_path+'participants.tsv', sep="\t")
    row_ = np.where(df['participant_id'] == 'sub-'+str(subject))[0][0]
    return df.iloc[row_]['line_noise']

def get_patient_coordinates(ch_names, ind_cortex, ind_subcortex, vhdr_file, BIDS_path):
    """
    for a given vhdr file, the respective BIDS path, and the used channel names of a BIDS run
    :return the coordinate file of the used channels 
        in shape (2): cortex; subcortex; fields might be empty (None if no cortex/subcortex channels are existent)
        appart from that the used fields are in numpy array field shape (num_coords, 3)
    """
    df = get_coords_df_from_vhdr(vhdr_file, BIDS_path)  # this dataframe contains all coordinates in this session
    coord_patient = np.empty(2, dtype=object)

    if ind_cortex is not None:
        coord_cortex = np.zeros([ind_cortex.shape[0], 3])
        for idx, ch_cortex in enumerate(np.array(ch_names)[ind_cortex]):
            coord_cortex[idx,:] = np.array(df[df['name']==ch_cortex].iloc[0][1:4]).astype('float')
        coord_patient[0] = coord_cortex
    
    if ind_subcortex is not None:
        coord_subcortex = np.zeros([ind_subcortex.shape[0], 3])
        for idx, ch_subcortex in enumerate(np.array(ch_names)[ind_subcortex]):
            coord_subcortex[idx,:] = np.array(df[df['name']==ch_subcortex].iloc[0][1:4]).astype('float')
        coord_patient[1] = coord_subcortex
        
    return coord_patient

def get_active_grid_points(sess_right, ind_label, ch_names, proj_matrix_run, grid_):
    """
    :param sess_right: boolean that determines if the session is left or right
    :ch_names : list from brainvision
    :proj_matrix_run : list: 0 - cortex; 1 - subcortex, projection array in shape grid_points X channels;
    returns: array in shape num grids points cortex_LEFT + subcortex_LEFT + cortex_RIGHT + subcortex_RIGHT 0/1 indication for 
        used interpolation or not
    """
    arr_act_grid_points = np.zeros([grid_[0].shape[1] + grid_[1].shape[1] + grid_[2].shape[1]+ grid_[3].shape[1]])
    label_channel = np.array(ch_names)[ind_label]
    Con_label = False; Ips_label = False

     #WRITE CONTRALATERAL DATA if the respective movement channel exists
    if sess_right is True:
        if len([ch for ch in label_channel if 'LEFT' in ch]) >0:
            Con_label =True
    elif sess_right is False:
        if len([ch for ch in label_channel if 'RIGHT' in ch]) >0:
            Con_label =True

    #WRITE IPSILATERAL DATA if the respective movement channel exists
    if sess_right is False:
        if len([ch for ch in label_channel if 'LEFT' in ch]) >0:
            Ips_label = True
    elif sess_right is True:
        if len([ch for ch in label_channel if 'RIGHT' in ch]) >0:
            Ips_label = True

    if Con_label is True and proj_matrix_run[0] is not None: 
        arr_act_grid_points[np.nonzero(np.sum(proj_matrix_run[0], axis=1))[0]] = 1
    if Ips_label is True and proj_matrix_run[0] is not None: 
        arr_act_grid_points[np.nonzero(np.sum(proj_matrix_run[0], axis=1))[0] + grid_[0].shape[1]] = 1
    if Con_label is True and proj_matrix_run[1] is not None: 
        arr_act_grid_points[np.nonzero(np.sum(proj_matrix_run[1], axis=1))[0] + grid_[0].shape[1]*2] = 1
    if Ips_label is True and proj_matrix_run[1] is not None: 
        arr_act_grid_points[np.nonzero(np.sum(proj_matrix_run[1], axis=1))[0] + grid_[0].shape[1]*2 + \
                            grid_[1].shape[1]] = 1

    return arr_act_grid_points

def get_sess_run_subject(vhdr_file):
    """
    Given a vhdr string return the including subject, run and session
    Args:
        vhdr_file (string): [description]
    
    Returns:
        subject, run, sess
    """
    
    subject = vhdr_file[vhdr_file.find('sub-')+4:vhdr_file.find('sub-')+7]
    
    str_run = vhdr_file[vhdr_file.find('run'):]
    run = str_run[str_run.find('-')+1:str_run.find('_')]
    
    str_sess = vhdr_file[vhdr_file.find('ses'):]
    sess = str_sess[str_sess.find('-')+1:str_sess.find('ieeg')-1]
    
    return subject, run, sess

def get_used_ch_idx(used_channels, ch_names_BV):
    """read from the provided list of used_channels the indices of channels in the channel names brainvision file

    Args:
        used_channels (list): used channels for a given location/type, e.g. cortical, subcortical, label
        ch_names_BV (list): channel names of brainvision file

    Returns:
        np array: indizes of channel names in ch_names_BV
    """
    used_idx = []
    for ch_track in used_channels:
        for ch_idx, ch in enumerate(ch_names_BV):
            if ch == ch_track:
                used_idx.append(ch_idx)
    return np.array(used_idx)

def get_dat_cortex_subcortex(bv_raw, ch_names, used_channels):
    """
    Data segemntation into cortex, subcortex, MOV and dat; returns also respective indizes of bv_raw
    :param bv_raw: raw np.array of Brainvision-read file
    :param ch_names
    """

    data_ = {
        "dat_cortex" : None, 
        "dat_subcortex" : None, 
        "dat_label" : None, 
        "ind_cortex" : used_channels['cortex'], 
        "ind_subcortex" : used_channels['subcortex'], 
        "ind_label" : used_channels['labels'], 
        "ind_dat" : None
    }

    if used_channels['cortex'] is not None:
        data_["dat_cortex"] = bv_raw[data_["ind_cortex"],:]

    if used_channels['subcortex'] is not None:
        data_["dat_subcortex"] = bv_raw[ data_["ind_subcortex"],:]

    if used_channels['labels'] is not None:
        data_["dat_label"] = bv_raw[data_["ind_label"] ,:]
        data_["ind_dat"] = np.arange(bv_raw.shape[0])[~np.isin(np.arange(bv_raw.shape[0]), data_["ind_label"])]
    
    return data_

def read_settings(file_name=None):
    if file_name is None:
        file_name='settings'
    
    with open('settings/'+file_name+'.json', 'rb') as f: 
        settings = json.load(f)
    return settings



def sess_right(sess):
    if 'right' in sess:
        sess_right = True
    else:
        sess_right = False
    return sess_right

def write_all_M1_channel_files():
    """

    Read all channels.tsv in the settings defined BIDS path, and write all all channels_M1.tsv files 
    --> copy all channel names from channel.tsv as name
    --> set targets to 'MOV' channels 
    --> rereference all to average 
    --> used all to 1 

    """

    settings = IO.read_settings()  # reads settings from settings/settings.json file in a dict 


    BIDS_channel_tsv_files = []
    for root, dirs, files in os.walk(settings['BIDS_path']):
        for file in files:
            if file.endswith("_channels.tsv"):
                ch_file = os.path.join(root, file)
                
                df_channel = pd.read_csv(ch_file, sep="\t")
                
                df = pd.DataFrame(np.nan, index=np.arange(len(list(df_channel['name']))), columns=['name', 'rereference', 'used', 'target'])

                df['used'] = 1

                df['name'] = list(df_channel['name'].copy(deep=True))

                ch_mov = [ch_idx for ch_idx, ch in enumerate(df_channel['name']) if ch.startswith('MOV')]
                target = np.zeros(len(list(df_channel['name'])))
                target[ch_mov] = 1
                df['target'] = target.astype(int)
                df['rereference'] = ['average']*len(list(df_channel['name']))

                df.to_csv(ch_file[:-12]+'channels_M1.tsv', sep='\t')

                BIDS_channel_tsv_files.append(ch_file)