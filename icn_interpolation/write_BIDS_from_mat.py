from bids import BIDSLayout
import settings
from settings import Settings
import os
from shutil import copyfile, copy
from mne_bids import write_raw_bids, make_bids_basename
import mne

import numpy as np
import scipy
import pybv


subject_path = '/Users/hi/Documents/workshop_ML/subjects/'


def get_channel_names(used_channels, rereferrencing_method, side):
    """

    :param used_channels:
    :param rereferrencing_method:
    :param side:
    :return:
    """
    channel_names = []
    stn_counter = 0; ecog_counter = 0
    if side[0] == 'left':
        side_ = 'LEFT_'
    else:
        side_ = 'RIGHT_'

    for ch in range(used_channels.shape[0]-2):
        if rereferrencing_method[used_channels][ch] == 'bipolar':
            channel_names.append('STN_'+side_+str(stn_counter))
            stn_counter+=1
        elif rereferrencing_method[used_channels][ch] == 'common_average':
            channel_names.append('ECOG_'+side_+str(ecog_counter))
            ecog_counter+=1
    channel_names.append('MOV_RIGHT')
    channel_names.append('MOV_LEFT')
    return channel_names



def read_raw_session(sess_file):
    """

    :param sess_file:
    :return:
    """
    dat = scipy.io.loadmat(sess_file)
    data = dat['data']
    rereferrencing_method = dat['rereferencing_method'][0]
    side = dat['side'][0]
    ecog = np.where(rereferrencing_method == 'common_average')[0]
    stn = np.where(rereferrencing_method == 'bipolar')[0]
    labels = np.where(rereferrencing_method == 'na')[0]
    if len(stn) !=0:
        ecog = np.concatenate((stn, ecog), axis=0)
    used_channels = np.concatenate((ecog, labels), axis=0)
    data = dat['data'][used_channels]
    channel_names = get_channel_names(used_channels, rereferrencing_method, side)
    return data, channel_names


def get_used_sessions(subject_path = settings.subject_path, list_DBS_folder=Settings.get_DBS_patients(settings.subject_path)):
    """
    :param subject_path:
    :param list_DBS_folder:
    :return: patient nested list with all used streams in form 'session_1_segment_1_N002_DBS4004_PD_UPMC.mat'
    """
    l_session_path = []
    for patient in list_DBS_folder:
        l_patient_path = []
        for file in os.listdir(os.path.join(subject_path,patient)):
            if file.endswith('.mat') == False:
                continue
            if file.endswith('UPMC.mat') == True:
                if 'stream' not in file:
                    l_patient_path.append(os.path.join(subject_path, patient, file))
        l_session_path.append(l_patient_path)
    return l_session_path


def write_BIDS():
    # write Brainvision format
    l_session_path = get_used_sessions()
    for patient_idx, patient_l in enumerate(l_session_path):

        for sess_idx, sess_path in enumerate(patient_l):
            data, channel_names = read_raw_session(l_session_path[patient_idx][sess_idx])
            pybv.write_brainvision(data, sfreq=1000, ch_names=channel_names, fname_base=str(sess_idx),
                                   folder_out='BrainVision',
                                   events=None, resolution=1e-7, scale_data=True,
                                   fmt='binary_float32', meas_date=None)

            # BIDS Filename
            if patient_idx < 10:
                subject_id = str('00') + str(patient_idx)
            else:
                subject_id = str('0') + str(patient_idx)

            raw = mne.io.read_raw_brainvision('BrainVision/' + str(sess_idx) + '.vhdr')
            run_ = sess_idx
            print(str(subject_id))

            if 'LEFT' in channel_names[0]:
                laterality = 'left'
            else:
                laterality = 'right'

            bids_basename = make_bids_basename(subject=str(subject_id),
                                               session=laterality, task='force',
                                               run=str(run_))
            # BIDS schreiben
            output_path = '/Users/hi/Documents/workshop_ML/thesis_plots/BIDS_new/'
            write_raw_bids(raw, bids_basename, output_path, event_id=0,
                           overwrite=True)


def copy_coord_from_BIDS_A_to_B(path_A, path_B):
    """
    simply copies all coordsystem.json and electrodes.tsv files from BIDS folder A to B
    :param path_A:
    :param path_B:
    :return: None
    """
    layout = BIDSLayout(settings.BIDS_path)
    subjects = layout.get_subjects()


    for patient_idx in range(len(subjects)):

        if patient_idx < 10:
            subject_id = str('00')+str(patient_idx)
        else:
            subject_id = str('0')+str(patient_idx)

        for lat in ['right', 'left']:
            path_A_subject = os.path.join(path_A+'sub-'+subject_id+'/ses-'+lat)
            if os.path.exists(path_A_subject):
                path_coord_sys = path_A_subject+'/eeg/'+'sub-'+subject_id+'_coordsystem.json'
                path_B_subject = os.path.join(path_B+'sub-'+subject_id+'/ses-'+lat)
                path_to_paste =  path_B_subject+'/eeg/'+'sub-'+subject_id+'_coordsystem.json'
                copy(path_coord_sys, path_to_paste)

                path_coord_sys = path_A_subject+'/eeg/'+'sub-'+subject_id+'_electrodes.tsv'
                path_B_subject = os.path.join(path_B+'sub-'+subject_id+'/ses-'+lat)
                path_to_paste =  path_B_subject+'/eeg/'+'sub-'+subject_id+'_electrodes.tsv'
                copy(path_coord_sys, path_to_paste)
