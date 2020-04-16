from bids import BIDSLayout
import sys
import settings
import pandas as pd
from settings import Settings
import os
import mne_bids
from shutil import copyfile, copy
from mne_bids import write_raw_bids, make_bids_basename
import mne
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import scipy
from scipy import signal
import pybv

def read_BIDS_file(file_path):
    """
    Read one run file from BIDS standard
    :param file_path: .vhdr file
    :return: raw dataset array, channel name array
    """
    bv_file = mne_bids.read.io.brainvision.read_raw_brainvision(file_path)
    bv_raw = bv_file.get_data()
    return bv_raw, bv_file.ch_names

def read_all_vhdr_filenames(BIDS_path):
    """
    :return: files: list of all vhdr file paths in BIDS_path
    """
    layout = BIDSLayout(BIDS_path)
    files = layout.get(extension='vhdr', return_type='filename')
    return files


def calc_band_filters(f_ranges, sample_rate, filter_len=1001, l_trans_bandwidth=4, h_trans_bandwidth=4, plot_=False):
    """
    This function returns for the given frequency band ranges filter coefficients with with length "filter_len"
    Thus the filters can be sequentially used for band power estimation
    """
    filter_fun = np.zeros([len(f_ranges), filter_len])

    for a, f_range in enumerate(f_ranges):
        h = mne.filter.create_filter(None, sample_rate, l_freq=f_range[0], h_freq=f_range[1], 
                            fir_design='firwin', verbose=plot_, l_trans_bandwidth=l_trans_bandwidth, 
                            h_trans_bandwidth=h_trans_bandwidth, filter_length='1000ms')
        if plot_ is True:              
            plot_filter(h,sample_rate)

        filter_fun[a, :] = h
    return filter_fun

def apply_filter(dat_, sample_rate, filter_fun, line_noise, seglengths):
    """
    For a given channel, apply 4 notch line filters and apply previously calculated filters
    return: variance in the given interval by seglength
    """
    dat_noth_filtered = mne.filter.notch_filter(x=dat_, Fs=sample_rate, trans_bandwidth=7,
            freqs=np.arange(line_noise, 4*line_noise, line_noise),
            fir_design='firwin', verbose=False, notch_widths=1,filter_length=dat_.shape[0]-1)

    filtered = np.zeros(filter_fun.shape[0])
    for filt in range(filter_fun.shape[0]):
        filtered[filt] = np.var(scipy.signal.convolve(filter_fun[filt,:], 
                                               dat_noth_filtered, mode='same')[-seglengths[filt]:])
    return filtered

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
    coord_path = os.path.join(BIDS_path, 'sub-'+ subject, 'ses-'+ sess, 'eeg', 'sub-'+ subject+ '_electrodes.tsv')
    df = pd.read_csv(coord_path, sep="\t")
    return df

def read_run_sampling_frequency(vhdr_file):
    """
    given a .eeg vhdr file, read the respective channel file and return the the sampling frequency for the first
    index, since all channels are throughout the run recorded with the same sampling frequency 
    """
    ch_file = vhdr_file[:-8]+'channels.tsv' # read out the channel, not eeg file (insted of eeg.eeg ending)
    df = pd.read_csv(ch_file, sep="\t")
    return df.iloc[0]['sampling_frequency']  

def read_line_noise(BIDS_path, subject):
    """
    return the line noise for a given subject (in shape '000') from participants.tsv
    """
    df = pd.read_csv(BIDS_path+'participants.tsv', sep="\t")
    row_ = np.where(df['participant_id'] == 'sub-'+str(subject))[0][0]
    return df.iloc[row_]['line_noise']

def get_patient_coordinates(ch_names, ind_ECOG, ind_STN, vhdr_file, BIDS_path):
    """
    for a given vhdr file, the respective BIDS path, and the used channel names of a BIDS run
    :return the coordinate file of the used channels 
        in shape (2): ECOG; STN; fiels might be empty (None if no ECOG/STN channels are existent)
        appart from that the used fields are in numpy array field shape (num_coords, 3)
    """
    df = get_coords_df_from_vhdr(vhdr_file, BIDS_path)  # this dataframe contains all coordinates in this session
    coord_ECOG = np.zeros([ind_ECOG.shape[0], 3])

    for idx, ch_ECOG in enumerate(np.array(ch_names)[ind_ECOG]):
        coord_ECOG[idx,:] = np.array(df[df['name']==ch_ECOG].iloc[0][1:4]).astype('float')

    if ind_STN.shape[0] !=0:
        coord_STN = np.zeros([ind_STN.shape[0], 3])
        for idx, ch_STN in enumerate(np.array(ch_names)[ind_STN]):
            coord_STN[idx,:] = np.array(df[df['name']==ch_STN].iloc[0][1:4]).astype('float')

    coord_patient = np.empty(2, dtype=object)
    
    if ind_ECOG.shape[0] !=0:
        coord_patient[0] = coord_ECOG
    if ind_STN.shape[0] !=0:
        coord_patient[1] = coord_STN
        
    return coord_patient

def calc_projection_matrix(coord_arr, grid_, sess_right, max_dist_ECOG = 20, max_dist_STN = 10):
    """
    calculates a projection matrix based on the used coord_arr of that BIDS run and the provided grid
    :param coord_arr: shape: (4) - ECOG LEFT; STN_LEFT; ECOG RIGHT; STN_RIGHT coordinate channel grid
    :param grid_: list with ecog left, stn left, ecog right, stn right coordinate grids
    :param max_dist_ECOG: float - defines interpolation parameter, for a given grid point take all 
        channels into account that have a euclidean distance to that channel in the max_dist_ECOG range
    :param max_dist_STN: float - defines interpolation parameter, for a given grid point take all 
        channels into account that have a euclidean distance to that channel in the max_dist_STN range
    :param sess_right - boolean - determines if electrodes had been recorded from left or right hemisphere
    :return: projection matrix array in shape 4: ECOG LEFT; STN_LEFT; ECOG RIGHT; STN_RIGHT
        here for each ECOG/STN LEFT/RIGHT location, the output has shape (grid_point X channel_in_location)
        for one grid point, the sum of all channel coefficients sums up to 1 
    """

    proj_matrix_run = np.empty(2, dtype=object)
    
    if sess_right is True: 
        grid_session = grid_[2:]
    else:
        grid_session = grid_[:2]
    
    
    for loc_, grid in enumerate(grid_session):
        
        if loc_ == 0:   # ECOG
            max_dist = max_dist_ECOG
        elif loc_ == 1:  # STN
            max_dist = max_dist_STN
            
        if coord_arr[loc_] is None:  #this checks if there are ECOG/STN channels in that run
            continue

        channels = coord_arr[loc_].shape[0]
        distance_matrix = np.zeros([grid.shape[1], channels])

        for project_point in range(grid.shape[1]):
            for channel in range(coord_arr[loc_].shape[0]):
                distance_matrix[project_point, channel] = \
                    np.linalg.norm(grid[:, project_point] - coord_arr[loc_][channel, :])
        
        
        proj_matrix = np.zeros(distance_matrix.shape)
        for grid_point in range(distance_matrix.shape[0]):
            used_channels = np.where(distance_matrix[grid_point, :] < max_dist)[0]

            rec_distances = distance_matrix[grid_point, used_channels]
            sum_distances = np.sum(1 / rec_distances)

            for ch_idx, used_channel in enumerate(used_channels):
                proj_matrix[grid_point, used_channel] = (1 / distance_matrix[grid_point, used_channel]) / sum_distances
        proj_matrix_run[loc_] = proj_matrix
        
    return proj_matrix_run

def get_projected_ECOG_STN_data(proj_matrix_run, sess_right, dat_ECOG=None, dat_STN=None):
    """
    :param proj_matrix_run - nparray that defines in shape (grid_points X channels) the projection weights
    :param sess_right - boolean - states if the session is left or right 
    :param dat_ECOG - nparray - of ECOG to project to grid 
    :param dat_STN - nparray - of STM to project to grid 
    :return projection ECOG data, projected STN data
    """
    proj_ECOG = None
    proj_STN = None

    if dat_ECOG is not None:
        proj_ECOG = proj_matrix_run[0] @ dat_ECOG
    if dat_STN is not None:
        proj_STN = proj_matrix_run[1] @ dat_STN

    return proj_ECOG, proj_STN

def get_active_grid_points(sess_right, ind_MOV, ch_names, proj_matrix_run, grid_):
    """
    :param sess_right: boolean that determines if the session is left or right
    :ch_names : list from brainvision
    :proj_matrix_run : list: 0 - ECOG; 1 - STN, projection array in shape grid_points X channels;
    returns: array in shape num grids points ECOG_LEFT + STN_LEFT + ECOG_RIGHT + STN_RIGHT 0/1 indication for 
        used interpolation or not
    """
    arr_act_grid_points = np.zeros([grid_[0].shape[1] + grid_[1].shape[1] + grid_[2].shape[1]+ grid_[3].shape[1]])
    mov_channel = np.array(ch_names)
    Con_label = False; Ips_label = False

     #WRITE CONTRALATERAL DATA if the respective movement channel exists
    if sess_right is True:
        if len([ch for ch in mov_channel if 'LEFT' in ch]) >0:
            Con_label =True
    elif sess_right is False:
        if len([ch for ch in mov_channel if 'RIGHT' in ch]) >0:
            Con_label =True



    #WRITE IPSILATERAL DATA if the respective movement channel exists
    if sess_right is False:
        if len([ch for ch in mov_channel if 'LEFT' in ch]) >0:
            Ips_label = True
    elif sess_right is True:
        if len([ch for ch in mov_channel if 'RIGHT' in ch]) >0:
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


def write_proj_data(ch_names, sess_right, dat_MOV, ind_MOV, proj_ECOG=None, proj_STN=None):
    """
    :param proj_ECOG - projected data on ECOG grid 
    """
    arr_all = np.empty([94, proj_ECOG.shape[1]])
    mov_channel = np.array(ch_names)[ind_MOV]

    Con_label = False; Ips_label = False
    
     #WRITE CONTRALATERAL DATA if the respective movement channel exists
    if sess_right is True:
        if len([ch for ch in mov_channel if 'LEFT' in ch]) >0:
            Con_label =True
            dat_mov_con = dat_MOV[[ch_idx for ch_idx, ch in enumerate(mov_channel) if 'LEFT' in ch][0],:]
            arr_all[:39,:] = proj_ECOG
            if proj_STN is not None:
                 arr_all[78:86,:] = proj_STN

    elif sess_right is False:
        if len([ch for ch in mov_channel if 'RIGHT' in ch]) >0:
            Con_label =True
            dat_mov_con = dat_MOV[[ch_idx for ch_idx, ch in enumerate(mov_channel) if 'RIGHT' in ch][0],:]
            arr_all[:39,:] = proj_ECOG
            if proj_STN is not None:
                 arr_all[78:86,:] = proj_STN


    #WRITE IPSILATERAL DATA if the respective movement channel exists
    if sess_right is False:
        if len([ch for ch in mov_channel if 'LEFT' in ch]) >0:
            Ips_label = True
            dat_mov_ips = dat_MOV[[ch_idx for ch_idx, ch in enumerate(mov_channel) if 'LEFT' in ch][0],:]

            arr_all[39:78,:] = proj_ECOG
            if proj_STN is not None:
                 arr_all[86:,:] = proj_STN
    elif sess_right is True:
        if len([ch for ch in mov_channel if 'RIGHT' in ch]) >0:
            Ips_label = True
            dat_mov_ips = dat_MOV[[ch_idx for ch_idx, ch in enumerate(mov_channel) if 'RIGHT' in ch][0],:]
            arr_all[39:78,:] = proj_ECOG
            if proj_STN is not None:
                 arr_all[86:,:] = proj_STN
    #ind_active_ = np.where(np.sum(arr_all, axis=1) != 0)
    return arr_all

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
    
def get_dat_ECOG_STN(bv_raw, ch_names):
    """
    Data segemntation into ECOG, STN, MOV and dat; returns also respective indizes of bv_raw
    :param bv_raw: raw np.array of Brainvision-read file
    :param ch_names
    """
    ind_ECOG = np.array([ch_idx for ch_idx, ch in enumerate(ch_names) if ch.startswith('ECOG')])
    ind_STN = np.array([ch_idx for ch_idx, ch in enumerate(ch_names) if ch.startswith('STN')])
    ind_MOV = np.array([ch_idx for ch_idx, ch in enumerate(ch_names) if ch.startswith('MOV') or ch.startswith('ANALOG')])
    ind_DAT = np.arange(bv_raw.shape[0])[~np.isin(np.arange(bv_raw.shape[0]), ind_MOV)]

    dat_ECOG = None; dat_STN = None
    if ind_ECOG.shape[0] !=0:
        dat_ECOG = bv_raw[ind_ECOG,:]
    if ind_STN.shape[0] !=0:
        dat_STN = bv_raw[ind_STN,:]
    dat_MOV = bv_raw[ind_MOV,:]
    
    return dat_ECOG, dat_STN, dat_MOV, ind_ECOG, ind_STN, ind_MOV, ind_DAT

def real_time_analysis(fs, fs_new, seglengths, f_ranges, grid_, downsample_idx, bv_raw, line_noise, \
                      sess_right, dat_ECOG, dat_STN, dat_MOV, ind_ECOG, ind_STN, ind_MOV, ind_DAT, \
                      filter_fun, proj_matrix_run, arr_act_grid_points):
    offset_start = int(seglengths[0] / (fs/fs_new))  # offset start is here the number of samples new_fs to skip 
    num_channels = ind_DAT.shape[0]
    num_grid_points = np.concatenate(grid_, axis=1).shape[1] # since grid_ is setup in ECOG left, STN left, ECOG right, STN right
    num_f_bands = len(f_ranges)

    rf_data = np.zeros([new_num_data_points-offset_start, num_channels, num_f_bands])  # raw frequency array
    rf_data_median = np.zeros([new_num_data_points-offset_start, num_channels, num_f_bands])
    pf_data = np.zeros([new_num_data_points-offset_start, num_grid_points, num_f_bands])  # projected 
    pf_data_median = np.zeros([new_num_data_points-offset_start, num_grid_points, num_f_bands])  # projected 
    mov_median = np.zeros([new_num_data_points, ind_MOV.shape[0]])
    new_idx = 0

    for c in range(downsample_idx.shape[0]):  
        print(str(np.round(c*(1/fs_new),2))+' s')
        if downsample_idx[c]<seglengths[0]:  # neccessary since downsample_idx starts with 0, wait till 1s for theta is over
            continue

        for ch in ind_DAT:    
            dat_ = bv_raw[ch, downsample_idx[c-offset_start]:downsample_idx[c]]
            dat_filt = apply_filter(dat_, sample_rate=fs, filter_fun=filter_fun, line_noise=line_noise, seglengths=seglengths)
            rf_data[new_idx,ch,:] = dat_filt

        #PROJECTION of RF_data to pf_data
        dat_ECOG = rf_data[new_idx, ind_ECOG,:]
        dat_STN = rf_data[new_idx, ind_STN,:]
        proj_ECOG, proj_STN = get_projected_ECOG_STN_data(proj_matrix_run, sess_right, dat_ECOG, dat_STN)
        pf_data[new_idx,:,:] = write_proj_data(ch_names, sess_right, dat_MOV, ind_MOV, proj_ECOG, proj_STN)

        #normalize acc. to Median of previous normalization samples
        if c<normalization_samples:
            if new_idx == 0:
                n_idx = 0
            else:
                n_idx = np.arange(0,new_idx,1)
        else:
            n_idx = np.arange(new_idx-normalization_samples, new_idx, 1)

        if new_idx == 0:
            rf_data_median[n_idx,:,:] = rf_data[n_idx,:,:]
            pf_data_median[n_idx,:,:] = pf_data[n_idx,:,:]
            mov_median[n_idx,:] = dat_MOV[:,n_idx]
        else:
            median_ = np.median(rf_data[n_idx,:,:], axis=0)
            rf_data_median[new_idx,:,:] = (rf_data[new_idx,:,:] - median_) / median_
            
            median_ = np.median(pf_data[n_idx,:,:][:,arr_act_grid_points>0,:], axis=0)
            pf_data_median[new_idx,arr_act_grid_points>0,:] = (pf_data[new_idx,arr_act_grid_points>0,:] - median_) / median_
            
            median_ = np.median(dat_MOV[:,n_idx], axis=1)
            mov_median[new_idx,:] = (dat_MOV[:,downsample_idx[c]] - median_) / median_
        new_idx += 1
    return rf_data_median, pf_data_median, mov_median
    
if __name__ == "__main__":
    
    BIDS_path = settings.BIDS_path
    files = read_all_vhdr_filenames(BIDS_path)
    vhdr_file = files[3]

    ecog_grid_left, ecog_grid_right, stn_grid_left, stn_grid_right = Settings.define_grid()
    grid_ = [ecog_grid_left, stn_grid_left, ecog_grid_right, stn_grid_right]

    bv_raw, ch_names = read_BIDS_file(vhdr_file)

    subject, run, sess = get_sess_run_subject(vhdr_file)
    if 'right' in sess:
        sess_right = True
    else:
        sess_right = False

    dat_ECOG, dat_STN, dat_MOV, ind_ECOG, ind_STN, ind_MOV, ind_DAT = get_dat_ECOG_STN(bv_raw, ch_names)

    coord_patient = get_patient_coordinates(ch_names, ind_ECOG, ind_STN, vhdr_file, BIDS_path)

    max_dist_ECOG = settings.max_dist_ECOG
    max_dist_STN = settings.max_dist_STN

    proj_matrix_run = calc_projection_matrix(coord_patient, grid_, sess_right, max_dist_ECOG, max_dist_STN)

    fs = read_run_sampling_frequency(vhdr_file)  # Hz NEEDS to be read out from the BIDS file
    f_ranges = settings.f_ranges
    line_noise = read_line_noise(BIDS_path,subject)
    normalization_time = settings.normalization_time
    fs_new = settings.fs_new 

    resample_factor = fs/fs_new
    seglengths = np.array([fs/1, fs/2, fs/2, fs/2, \
              fs/2, fs/10, fs/10, fs/10]).astype(int)

    normalization_samples = normalization_time*fs_new
    new_num_data_points = int((bv_raw.shape[1]/fs)*fs_new)

    # downsample_idx states the original brainvision sample indexes are used
    downsample_idx = (np.arange(0,new_num_data_points,1)*fs/fs_new).astype(int)

    filter_fun = calc_band_filters(f_ranges, fs)

    offset_start = int(seglengths[0] / (fs/fs_new))

    arr_act_grid_points = get_active_grid_points(sess_right, ind_MOV, ch_names, proj_matrix_run, grid_)

    rf_data_median, pf_data_median, mov_median = real_time_analysis(fs, fs_new, seglengths, f_ranges, grid_, downsample_idx, bv_raw, line_noise, \
                      sess_right, dat_ECOG, dat_STN, dat_MOV, ind_ECOG, ind_STN, ind_MOV, ind_DAT, \
                      filter_fun, proj_matrix_run, arr_act_grid_points)