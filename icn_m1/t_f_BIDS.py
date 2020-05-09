#  this function gets a vhdr file and read the BIDS run
# it estimates a T-F transform for every channel
# this is independent of the used laterality and channel

import json
import numpy as np
import mne_bids
import os
from mne.time_frequency import tfr_morlet
from bids import BIDSLayout
from mne.decoding import TimeFrequency
from matplotlib import pyplot as plt
from scipy import stats


# read a raw BIDS file
def read_BIDS_file(vhdr_file):
    """
    Read one run file from BIDS standard
    :param vhdr_file:
    :return: raw dataset array, channel name array
    """
    bv_file = mne_bids.read.io.brainvision.read_raw_brainvision(vhdr_file)
    bv_raw = bv_file.get_data()
    return bv_raw, bv_file.ch_names

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
    sess = str_sess[str_sess.find('-')+1:str_sess.find('eeg')-1]

    return subject, run, sess

def read_all_vhdr_filenames(BIDS_path):
    """
    :return: files: list of all vhdr file paths in BIDS_path
    """
    layout = BIDSLayout(BIDS_path)
    files = layout.get(extension='vhdr', return_type='filename')
    return files

def write_TF_from_BIDS(vhdr_file, out_path='C:\\Users\\ICN_admin\\Documents\\write_TF\\out'):
    bv_raw, ch_names = read_BIDS_file(vhdr_file)

    TF_ = TimeFrequency(freqs=np.arange(1,200, 1), sfreq=1000, n_cycles=5, method='morlet', output='power')
    run_TF = TF_.transform(bv_raw[:-2,:])
    dat_z = stats.zscore(run_TF, axis=2)

    if 'RIGHT' in ch_names[0] and 'RIGHT' in ch_names[-1]:
        mov_ips = bv_raw[-1,:]
        mov_con = bv_raw[-2,:]
    if 'LEFT' in ch_names[0] and 'LEFT' in ch_names[-1]:
        mov_ips = bv_raw[-1,:]
        mov_con = bv_raw[-2,:]
    if 'RIGHT' in ch_names[0] and 'RIGHT' in ch_names[-2]:
        mov_ips = bv_raw[-2,:]
        mov_con = bv_raw[-1,:]
    if 'LEFT' in ch_names[0] and 'LEFT' in ch_names[-2]:
        mov_ips = bv_raw[-2,:]
        mov_con = bv_raw[-1,:]

    dict_out = {
        "dat_z" : dat_z.tolist(),
        "ch_labels" : ch_names,
        "mov_con" : mov_con.tolist(),
        "mov_ips" : mov_ips.tolist()
    }

    subject, run, sess = get_sess_run_subject(vhdr_file)

    with open(os.path.join(out_path, 'sub_'+subject+'_ses-'+sess+'_run_'+run+'_tf.json'), 'w') as fp:
        json.dump(dict_out, fp)
