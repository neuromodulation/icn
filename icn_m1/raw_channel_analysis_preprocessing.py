import os
import numpy as np 
import pandas as pd
from scipy import stats, signal
import mne
from bids import BIDSLayout
import mne_bids
import settings
import json
from coordinates_io import BIDS_coord
from sklearn import linear_model
import multiprocessing
from sklearn.model_selection import cross_val_score

def read_BIDS_file(vhdr_file):
    """
    Read one run file from BIDS standard
    :param vhdr_file: 
    :return: raw dataset array, channel name array
    """
    bv_file = mne_bids.read.io.brainvision.read_raw_brainvision(vhdr_file)
    bv_raw = bv_file.get_data()
    return bv_raw, bv_file.ch_names

def read_all_vhdr_filenames(BIDS_path):
    """
    :return: files: list of all vhdr file paths in BIDS_path
    """
    layout = BIDSLayout(BIDS_path)
    files = layout.get(extension='vhdr', return_type='filename')
    return files

def get_line_noise(vhdr_file):
    """Given a vhdr file, the path is altered in order to read the JSON description file
    
    Args:
        vhdr_file ([type]): [description]
    
    Returns:
        int: Power Line Frequency
    """
    json_run_ = vhdr_file[:-4] + 'json'
    with open(json_run_, 'r') as fp:
        json_run_descr = json.load(fp)
    return int(json_run_descr['PowerLineFrequency'])

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

def running_z_score(x_filtered, z_score_running_interval):
    """

    :param x_filtered
    :param z_score_running_interval
    :return: z-scored stream wrt consecutive time interval
    """
    x_filtered_zscored = np.zeros([x_filtered.shape[0], x_filtered.shape[1], x_filtered.shape[2] - z_score_running_interval])
    for band in range(x_filtered.shape[0]):
        for ch in range(x_filtered.shape[1]):
            for time in np.arange(z_score_running_interval, x_filtered.shape[2], 1):
                running_mean = np.mean(x_filtered[band, ch, (time - z_score_running_interval):time])
                running_std = np.std(x_filtered[band, ch, (time - z_score_running_interval):time])
                x_filtered_zscored[band, ch, time - z_score_running_interval] = \
                    (x_filtered[band, ch, time] - running_mean) / running_std
    return x_filtered_zscored

def running_zscore_label(mov_label, z_score_running_interval):
    """

       :param mov_label
       :param z_score_running_interval
       :return: z-scored stream wrt consecutive time interval
       """
    mov_label_zscored = np.zeros([mov_label.shape[0], mov_label.shape[1] - z_score_running_interval])
    for ch in range(mov_label.shape[0]):
        for time in np.arange(z_score_running_interval, mov_label.shape[1], 1):
            running_mean = np.mean(mov_label[ch, (time - z_score_running_interval):time])
            running_std = np.std(mov_label[ch, (time - z_score_running_interval):time])
            mov_label_zscored[ch, time - z_score_running_interval] = \
                (mov_label[ch, time] - running_mean) / running_std
    return mov_label_zscored

def z_score_offline(x_filtered):
    """

    :param x_filtered
    :return: simple "offline" z-score for quicker analysis
    """
    x_filtered_zscored = np.zeros(x_filtered.shape)
    for band in range(x_filtered.shape[0]):
        for ch in range(x_filtered.shape[1]):
            x_filtered_zscored[band, ch, :] = stats.zscore(x_filtered[band, ch, :])
    return x_filtered_zscored

def z_score_offline_label(mov_label):
    """

    :param mov_label
    :return: simple "offline" z-score for quicker analysis
    """
    mov_label_zscored = np.zeros(mov_label.shape)
    for ch in range(mov_label.shape[0]):
        mov_label_zscored[ch, :] = stats.zscore(mov_label[ch, :])
    return mov_label_zscored


def t_f_transform(x, sample_rate, f_ranges, line_noise):
    """
    calculate time frequency transform with mne filter function
    """
    filtered_x = []

    for f_range in f_ranges:
        if line_noise in np.arange(f_range[0], f_range[1], 1):
            #do line noise filtering

            x = mne.filter.notch_filter(x=x, Fs=sample_rate, 
                freqs=np.arange(line_noise, 4*line_noise, line_noise), 
                fir_design='firwin', verbose=False, notch_widths=2)

        h = mne.filter.create_filter(x, sample_rate, l_freq=f_range[0], h_freq=f_range[1], \
                                     fir_design='firwin', verbose=False, l_trans_bandwidth=2, h_trans_bandwidth=2)
        filtered_x.append(np.convolve(h, x, mode='same'))
 
    return np.array(filtered_x)


def transform_channels(bv_raw, line_noise):
    """
    calculate t-f-transform for every channel
    :param bv_raw: Raw (channel x time) datastream
    :return: t-f transformed array in shape (len(f_ranges), channels, time)
    """
    x_filtered = np.zeros([len(settings.f_ranges), bv_raw.shape[0], bv_raw.shape[1]])
    for ch in range(bv_raw.shape[0]):
        x_filtered[:, ch, :] = t_f_transform(bv_raw[ch, :], settings.sample_rate, settings.f_ranges, line_noise)
    return x_filtered


def calc_running_var(x_filtered_zscored, mov_label_zscored, var_interval=settings.var_rolling_window):
    """
    Given the filtered and z-scored data, apply a rolling variance winow
    :param x_filtered_zscored
    :param mov_label_zscored
    :param var_interval time window in which the variance is acquired
    :return: datastream and movement adapted arrays
    """
    stream_roll = np.array(pd.Series(x_filtered_zscored[0, 0, :]).rolling(window=var_interval).var())
    stream_roll = stream_roll[~np.isnan(stream_roll)]
    time_series_length = stream_roll.shape[0]

    x_filtered_zscored_var = np.zeros([x_filtered_zscored.shape[0], x_filtered_zscored.shape[1], time_series_length])

    for f in range(len(settings.f_ranges)):
        for ch in range(x_filtered_zscored.shape[1]):
            stream_roll = np.array(pd.Series(x_filtered_zscored[f, ch, :]).rolling(window=var_interval).var())
            if stream_roll[~np.isnan(stream_roll)].shape[0] == 0:
                x_filtered_zscored_var[f, ch, :] = np.zeros(x_filtered_zscored_var[f, ch, :].shape[0])
            else:
                x_filtered_zscored_var[f, ch, :] = stream_roll[~np.isnan(stream_roll)]
    # change the label vector too
    return x_filtered_zscored_var, mov_label_zscored[:, (x_filtered_zscored.shape[2] - time_series_length):]

def resample(vhdr_file, ch_names, x_filtered_zscored, mov_label_zscored):
    """Data and mov vector is resampled, assumption here: all channels have the same sampling sampling frequency
    
    Args:
        vhdr_file (): [description]
        ch_names ([type]): [description]
        x_filtered_zscored ([type]): [description]
        mov_label_zscored ([type]): [description]
    """

    #sub-002_ses-right_task-force_run-0_channels.tsv
    #sub-002_ses-right_task-force_run-0_eeg.vhdr

    fs_new = settings.resampling_rate 
    ch_file = vhdr_file[:-8] + 'channels.tsv'  # the channel file name has the same path/structure as the vhdr file
    df = pd.read_csv(ch_file, sep="\t")
    
    ch_name = ch_names[0]
    ind_ch = np.where(df['name'] == ch_name)[0][0]  # read out the dataframes channel names frequency, here implementation: same fs for all channels in one run
    fs = df['sampling_frequency'][ind_ch]

    dat_points = x_filtered_zscored.shape[2]
    new_num_data_points = int((dat_points/fs)*fs_new)
    dat_resampled = signal.resample(x_filtered_zscored, num=new_num_data_points, axis=2)
    mov_resampled = signal.resample(mov_label_zscored, num=new_num_data_points, axis=1)

    return dat_resampled, mov_resampled

def write_out_raw(vhdr_file, folder_out=settings.out_path_folder, test_LM=False, resampling=True):
    """
    Multiprocessing "Pool" function to interpolate raw file from vhdr_file write to out_path
    :param vhdr_file: raw .vhdr file
    :param out_path_folder
    """

    subject, run, sess = get_sess_run_subject(vhdr_file)

    bv_raw, ch_names = read_BIDS_file(vhdr_file)
    ind_mov = [ch_idx for ch_idx, ch in enumerate(ch_names) if ch.startswith('MOV') or ch.startswith('ANALOG')]
    
    # approach only indexing ECOG named channels
    #ind_dat = [ch_idx for ch_idx, ch in enumerate(ch_names) if ch.startswith('ECOG') or ch.startswith('ANALOG')]
    ind_dat = np.arange(bv_raw.shape[0])[~np.isin(np.arange(bv_raw.shape[0]), ind_mov)]

    mov_label = bv_raw[ind_mov, :] 

    line_noise = get_line_noise(vhdr_file)
    
    #bug fix for now, since I don't see a way to insert writing the line noise parameter in write_brainvision (pybv) for write_raw_bids (mne_bids)
    if subject != '016':
        line_noise = 60

    x_filtered = transform_channels(bv_raw[ind_dat, :], line_noise)

    # proxy for offline data analysis
    # it might be that there are NaN values due to no data stream...
    x_filtered_zscored = np.nan_to_num(z_score_offline(x_filtered))
    mov_label_zscored = np.nan_to_num(z_score_offline_label(mov_label))
    
    x_filtered_zscored, mov_label_zscored = calc_running_var(x_filtered_zscored, mov_label_zscored, var_interval=1000)
    x_filtered_zscored = np.clip(x_filtered_zscored, settings.clip_low, settings.clip_high)

    if test_LM is True:
        for ch in range(bv_raw[ind_dat, :].shape[0]):
            print(np.mean(cross_val_score(linear_model.LinearRegression(), x_filtered_zscored[:,ch,:].T, mov_label_zscored[0,:], cv=5)))

    if resampling is True:
        x_filtered_zscored, mov_label_zscored = resample(vhdr_file, ch_names, x_filtered_zscored, mov_label_zscored)

    dict_ = {
        "data": x_filtered_zscored.tolist(),
        "true_movements": mov_label_zscored.tolist(),
        "ch_names": ch_names, 
        "coords": BIDS_coord.get_coord_from_vhdr(settings.BIDS_path, vhdr_file), 
        "subject": subject, 
        "run": run, 
        "sess": sess
    }


    outpath_file = os.path.join(folder_out, 'raw_sub_' + subject + '_run_' + run + '_sess_' + sess + '.json')
    with open(outpath_file, 'w') as fp:
        json.dump(dict_, fp)

def start_pool_all_runs():

    vhdr_files = read_all_vhdr_filenames(settings.BIDS_path)


    write_out_raw(vhdr_files[0], folder_out="/home/icn/Documents/raw_out/raw_runs_non_downsampeled/", test_LM=False, resampling=False)
    write_out_raw(vhdr_files[1], folder_out="/home/icn/Documents/raw_out/raw_runs_non_downsampeled/", test_LM=False, resampling=False)
    write_out_raw(vhdr_files[2], folder_out="/home/icn/Documents/raw_out/raw_runs_non_downsampeled/", test_LM=False, resampling=False)
    write_out_raw(vhdr_files[3], folder_out="/home/icn/Documents/raw_out/raw_runs_non_downsampeled/", test_LM=False, resampling=False)

    vhdr_files = read_all_vhdr_filenames(settings.BIDS_path)
    pool = multiprocessing.Pool()
    pool.map(write_out_raw, vhdr_files)

if __name__ == "__main__":

    vhdr_files = read_all_vhdr_filenames(settings.BIDS_path)
    write_out_raw(vhdr_files[36], test_LM=True)
    start_pool_all_runs()