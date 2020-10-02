import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
import scipy
import mne
import os
import pandas as pd
import numpy as np
import mne
import scipy
import pickle


BIDS_PATH = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\"
PATH_OUT = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\Combined_runs\\"

def get_all_files_of_type(BIDS_PATH, type_='.vhdr'):
    """
    Given a BIDS path return all vhdr file paths without BIDS_Layout
    Args:
        BIDS_path (string)
    Returns:
        vhdr_files (list)
    """
    vhdr_files = []
    for root, dirs, files in os.walk(BIDS_PATH):
        for file in files:
            if file.endswith(type_):
                vhdr_files.append(os.path.join(root, file))
    return vhdr_files

def get_mov_dict(raw, sess):

    """
    Given a brainvision raw object, and the respective session (left / right),
    return the contra and or ipsilateral label
    return None, if label does not exist
    """

    ind_RIGHT = [ch_idx for ch_idx, ch in enumerate(raw.ch_names) if 'RIGHT_CLEAN' in ch]
    ind_LEFT = [ch_idx for ch_idx, ch in enumerate(raw.ch_names) if 'LEFT_CLEAN' in ch]

    dict_mov = {"mov_con":None, "mov_ips":None}

    if (len(ind_LEFT) == 1 and "right" in sess):
        dict_mov["mov_con"] = raw.get_data()[ind_LEFT, :][0,:]
    if (len(ind_RIGHT) == 1 and "left" in sess):
        dict_mov["mov_con"] = raw.get_data()[ind_RIGHT, :][0,:]
    if (len(ind_RIGHT) == 1 and "right" in sess):
        dict_mov["mov_ips"] = raw.get_data()[ind_RIGHT, :][0,:]
    if (len(ind_LEFT) == 1 and "left" in sess):
        dict_mov["mov_ips"] = raw.get_data()[ind_lEFT, :][0,:]
    return dict_mov

# get subject
def get_subject_tsv_runs(subject_id):
    """
    Parameter subject_id in form "000" and return available .tsv, run file names and coordinate df
    """
    ses_ = []
    if os.path.exists(os.path.join(BIDS_PATH, 'sub-'+subject_id, 'ses-right')) is True:
        ses_.append('right')
    if os.path.exists(os.path.join(BIDS_PATH, 'sub-'+subject_id, 'ses-left')) is True:
        ses_.append('left')

    tsv_files = get_all_files_of_type(BIDS_PATH, '.tsv')

    return tsv_files, ses_

def get_combined_runs(tsv_files, ses_):
    """
    Write out data, con. and ips. movement and coordinates in dictionary for each channel
    """
    dict_ch = {}


    for ses in ses_:
        elec_path = os.path.join(BIDS_PATH, 'sub-'+subject_id, 'ses-'+ses,'ieeg','sub-'+subject_id+'_electrodes.tsv')
        df_elec = pd.read_csv(elec_path, sep="\t")
        runs_ = [file for file in tsv_files if 'sub-'+subject_id in file and 'ses-'+ses in file and file.endswith('channels.tsv')]
        for ch in df_elec['name']:
            start = 0
            for run in runs_:
                df_run = pd.read_csv(run, sep="\t")
                if ch in list(df_run['name']):
                    ind_data = np.where(df_run['name'] == ch)[0][0]
                    run_number = run[run.find('run-')+4:run.find('_channels')] # is a string
                    raw = mne.io.read_raw_brainvision(run[:-12]+"ieeg.vhdr")
                    if start == 0:
                        start = 1
                        ch_dat = raw.get_data()[ind_data, :]
                        mov_dict = get_mov_dict(raw, "right")
                        if mov_dict["mov_con"] is not None:
                            mov_con = mov_dict["mov_con"]
                        if mov_dict["mov_ips"] is not None:
                            mov_ips = mov_dict["mov_ips"]
                    else:
                        ch_dat = np.concatenate((ch_dat, raw.get_data()[ind_data, :]), axis=0)
                        mov_dict = get_mov_dict(raw, "right")
                        if mov_dict["mov_con"] is not None:
                            mov_con = np.concatenate((mov_con, mov_dict["mov_con"]), axis=0)

                        if mov_dict["mov_ips"] is not None:
                            mov_ips = np.concatenate((mov_ips, mov_dict["mov_ips"]), axis=0)
            if start != 0:
                dict_ch[ch] = {
                    "data": ch_dat,
                    "mov_con": mov_con,
                    "mov_ips": mov_ips,
                    "coords": np.array(df_elec.loc[np.where(df_elec['name'] == ch)[0][0]][1:4], float)
                }
    return dict_ch

if __name__ == '__main__':

    for sub_idx  in range(16):
        print(sub_idx)
        if sub_idx<10:
            subject_id = '00' + str(sub_idx)
        else:
            subject_id = '0' + str(sub_idx)
        tsv_files, ses_ = get_subject_tsv_runs(subject_id)
        dict_ch = get_combined_runs(tsv_files, ses_)
        with open(os.path.join(PATH_OUT, 'sub_'+subject_id+'_comb.p'), 'wb') as handle:
            pickle.dump(dict_ch, handle, protocol=pickle.HIGHEST_PROTOCOL)
