import mne
import mne_bids
from mne_bids import write_raw_bids, make_bids_basename
from mne_bids.datasets import fetch_faces_data
from mne_bids.utils import print_dir_tree
from matplotlib import pyplot as plt
import numpy as np
import pybv
from scipy import signal, stats
from bids import BIDSLayout
import pandas as pd
import multiprocessing
import pickle
import os
import settings
from settings import Settings

coord_arr, coord_arr_names = Settings.read_BIDS_coordinates()
ecog_grid_left, ecog_grid_right, stn_grid_left, stn_grid_right = Settings.define_grid()
grid_ = [ecog_grid_left, stn_grid_left, ecog_grid_right, stn_grid_right]


def read_BIDS_file(file_path):
    """
    Read one run file from BIDS standard
    :param file_path: .vhdr file
    :return: raw dataset array, channel name array
    """
    bv_file = mne_bids.read.io.brainvision.read_raw_brainvision(file_path)
    bv_raw = bv_file.get_data()
    return bv_raw, bv_file.ch_names


def t_f_transform(x, sample_rate, f_ranges):
    """
    :param x: given array in form [f_range, time]
    :param sample_rate
    :param f_ranges
    :return: list of filtered stream for all frequency bands
    """

    filtered_x = []
    nyq_rate = sample_rate / 2
    width = 1.0 / nyq_rate
    N, beta = signal.kaiserord(settings.ripple_db, width)
    noise_ = np.arange(settings.line_noise[0], settings.line_noise[1] + 1, 1)

    for f_range in f_ranges:

        cutoff_hz = f_range  #BP Filter range
        f_range = np.arange(f_range[0], f_range[1] + 1, 1)
        taps = signal.firwin(N, np.array(cutoff_hz) / nyq_rate, window = ('kaiser', beta), pass_zero=False)


        if np.intersect1d(noise_, f_range).shape[0] != 0:
            #do line noise filtering
            taps_ = signal.firwin(N, np.array(settings.line_noise) / nyq_rate, window=('kaiser', beta))
            x = signal.lfilter(taps_, 1.0, x)

        time_filtered = signal.lfilter(taps, 1.0, x)
        filtered_x.append(time_filtered)
    return np.array(filtered_x)


def transform_channels(bv_raw):
    """
    calculate t-f-transform for every channel
    :param bv_raw: Raw (channel x time) datastream
    :return: t-f transformed array in shape (len(f_ranges), channels, time)
    """
    x_filtered = np.zeros([len(settings.f_ranges), bv_raw.shape[0] - 2, bv_raw.shape[1]])
    for ch in range(bv_raw.shape[0] - 2):
        x_filtered[:, ch, :] = t_f_transform(bv_raw[ch, :], settings.sample_rate, settings.f_ranges)
    return x_filtered


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


def calc_running_var(x_filtered_zscored, mov_label_zscored):
    """
    Given the filtered and z-scored data, apply a rolling variance winow
    :param x_filtered_zscored
    :param mov_label_zscored
    :return: datastream and movement adapted arrays
    """
    stream_roll = np.array(pd.Series(x_filtered_zscored[0, 0, :]).rolling(window=500).var())
    stream_roll = stream_roll[~np.isnan(stream_roll)]
    time_series_length = stream_roll.shape[0]

    x_filtered_zscored_var = np.zeros([x_filtered_zscored.shape[0], x_filtered_zscored.shape[1], time_series_length])

    for f in range(len(settings.f_ranges)):
        for ch in range(x_filtered_zscored.shape[1]):
            stream_roll = np.array(pd.Series(x_filtered_zscored[0, 0, :]).rolling(window=settings.var_rolling_window).var())
            x_filtered_zscored_var[f, ch, :] = stream_roll[~np.isnan(stream_roll)]
    # change the label vector too
    return x_filtered_zscored_var, mov_label_zscored[:, (x_filtered_zscored.shape[2] - time_series_length):]


def get_same_ECOG_indices(ch_names, coord_arr_names, subject_idx, left=True):
    """
    This function checks given the bv_raw channel names array, and the names given from the coordinate file, which coordinates are used
    This is neccessary, since in different runs, different number of strips can be active

    :param ch_names: channels names read from brainvision
    :param coord_arr_names: names given by the electrodes tsv file
    :param subject_idx: BIDS patient index
    :param left: boolean
    :return: ECOG indices which should be used when indexing the distance matrix_arr
    """

    if left is True:
        lat_ = 0
    else:
        lat_ = 1
    ch_names_ecog = [i for i in ch_names if i.startswith('ECOG_')]
    coord_names_ecog = [i for i in coord_arr_names[subject_idx][lat_] if i.startswith('ECOG_')]
    ch_used_in_run = np.where(np.in1d(ch_names_ecog, coord_names_ecog))[0]

    return ch_used_in_run


def calc_projection_matrix(subject_idx, coord_arr, grid_):
    """
    calculate a distance array in shape (grid_point, channel)
    :param subject_idx: subject
    :param coord_arr: (len(subject), 4) - ecog left, stn left, ecog right, stn right coordinate channel grid
    :param ecog left, stn left, ecog right, stn right coordinate grid
    :return: projection matrix array for the respective subject in shape (grid_point, channel)
    """

    matrix_arr = np.empty(4, dtype=object)

    for grid_idx, grid in enumerate(grid_):

        if coord_arr[subject_idx, grid_idx] is None:
            continue

        channels = coord_arr[subject_idx, grid_idx].shape[0]
        projection_matrix = np.zeros([grid.shape[1], channels])

        for project_point in range(grid.shape[1]):
            for channel in range(coord_arr[subject_idx, grid_idx].shape[0]):
                projection_matrix[project_point, channel] = \
                    np.linalg.norm(grid[:, project_point] - coord_arr[subject_idx, grid_idx][channel, :])
        matrix_arr[grid_idx] = projection_matrix
    return matrix_arr

def interpolate_stream(x_filtered_zscored, mov_label_zscored, matrix_arr_all, subject_idx, ch_names, int_distance_ecog=settings.int_distance_ecog, int_distance_stn=settings.int_distance_stn):
    """
    Given a channel datastream, this function implements the interpolation to the Settings defined Grid.
    Here contra -and ipsilateral grid points are separated
    For every run the projection is saved on the contra -and ipsilateral hemisphere
    The label vector is switched respectively
    :param x_filtered_zscored
    :param mov_label_zscored
    :param matrix_arr_all: distance array (grid_point, channel)
    :param subject_idx
    :param ch_names: read out raw channel names from BIDS
    :param int_distance_ecog
    :param int_distance_stn
    :return: int_data: interpolated array (94) includes NaN if no interpolation was performed
    :return: label_mov: label with 0-contralateral, 1-ipsilateral
    :return: act_grid_points: active interpolated grid points in shape 94
    """
    int_data = np.empty(grid_[0].shape[1] * 2 + grid_[1].shape[1] * 2, dtype=object)  # 94 shape

    act_grid_points = np.zeros(grid_[0].shape[1] * 2 + grid_[1].shape[1] * 2)
    label_mov = np.zeros(mov_label_zscored.shape)  # 0: contralateral, 1: ipsilateral

    if 'RIGHT' in ch_names[0]:
        #  right: here firstly defined as grid points 0:39, later the label is changed respectively
        offset_ecog = 0
        offset_stn = 78
    else:
        # left: here firstly defined as ipsilateral
        offset_ecog = 39
        offset_stn = 86

    for index in range(4):  # [ecog_grid_left, stn_grid_left, ecog_grid_right, stn_grid_right]

        if ('RIGHT' in ch_names[0]) & (index == 0 or index == 1):
            continue
        if ('LEFT' in ch_names[0]) & (index == 2 or index == 3):
            continue

        if index == 0 or index == 2:
            int_distance = int_distance_ecog
            if index == 0:
                ch_used_in_run = get_same_ECOG_indices(ch_names, coord_arr_names, subject_idx, left=True)
            else:
                ch_used_in_run = get_same_ECOG_indices(ch_names, coord_arr_names, subject_idx, left=False)
            matrix_arr = matrix_arr_all[index][:,ch_used_in_run]
        elif index == 1 or index == 3:
            int_distance = int_distance_stn
            matrix_arr = matrix_arr_all[index]


        for grid_point in range(grid_[index].shape[1]):
            used_channels = np.where(matrix_arr[grid_point, :] < int_distance)[0]
            if used_channels.shape[0] == 0:
                continue
            rec_distances = matrix_arr[grid_point, used_channels]
            sum_distances = np.sum(1 / rec_distances)
            first_ch = 0
            for ch_idx, used_channel in enumerate(used_channels):
                if first_ch == 0:
                    first_ch = 1
                    running_stream = (1 / matrix_arr[grid_point, used_channel]) * \
                                     x_filtered_zscored[:, used_channel, :] / sum_distances
                else:
                    running_stream += (1 / matrix_arr[grid_point, used_channel]) * \
                                      x_filtered_zscored[:, used_channel, :] / sum_distances

            if index == 0 or index == 2:
                int_data[grid_point + offset_ecog] = running_stream
                act_grid_points[grid_point + offset_ecog] = 1
            elif index == 1 or index == 3:
                int_data[grid_point + offset_stn] = running_stream
                act_grid_points[grid_point + offset_stn] = 1

    if offset_ecog == 0:  # right
        # copy data from right to left
        int_data[39:78] = int_data[:39];
        act_grid_points[39:78] = act_grid_points[:39]
        int_data[86:94] = int_data[78:86];
        act_grid_points[86:94] = act_grid_points[78:86]
        label_mov[0, :] = mov_label_zscored[1, :]  # contralateral is here left, since the strip lies on the right side
        label_mov[1, :] = mov_label_zscored[0, :]
    else:
        int_data[:39] = int_data[39:78];
        act_grid_points[:39] = act_grid_points[39:78];
        int_data[78:86] = int_data[86:94];
        act_grid_points[78:86] = act_grid_points[86:94];
        label_mov = mov_label_zscored  #  contralateral is here right, since strip is left located

    return int_data, label_mov, act_grid_points


def get_name(str_):
    """
    Read the given .vhdr path and return output path including patient_idx, run and respective side
    :param str_: vhdr filepath
    :return: subject_idx
    :return: string containing file name in form sub-000_run-0_right
    """

    run_ = str_[str_.find('run-'):str_.find('_eeg.vhdr')]
    subject = 'sub-'+str_.partition('sub-')[2][:3]
    if 'right' in str_.partition('ses-')[2]:
        name = subject+'_'+run_+str('_')+'right'
    else:
        name = subject+'_'+run_+str('_')+'left'
    return int(subject[4:]), name


def write_and_interpolate_vhdr(file_path):
    """
    Multiprocessing "Pool" function to interpolate raw file from file_path write to out_path
    :param file_path: raw .vhdr file
    :param out_path_folder
    """
    bv_raw, ch_names = read_BIDS_file(file_path)
    mov_label = bv_raw[-2:, :]  # last two elements are labels
    x_filtered = transform_channels(bv_raw)

    # proxy for offline data analysis
    x_filtered_zscored = z_score_offline(x_filtered)
    mov_label_zscored = z_score_offline_label(mov_label)

    # online real time z-scoring
    # x_filtered_zscored = running_z_score(x_filtered, z_score_running_interval)
    # mov_label_zscored = running_zscore_label(mov_label, z_score_running_interval)  # does not yield desired results...

    # clipping for artifact rejection
    x_filtered_zscored = np.clip(x_filtered_zscored, settings.clip_low, settings.clip_high)
    x_filtered_zscored, mov_label_zscored = calc_running_var(x_filtered_zscored, mov_label_zscored)

    subject_idx, file_name_out = get_name(file_path)

    matrix_arr = calc_projection_matrix(subject_idx, coord_arr, grid_)

    int_data, label_mov, act_grid_points = interpolate_stream(x_filtered_zscored, mov_label_zscored,
                                                              matrix_arr, subject_idx, ch_names,
                                                              int_distance_ecog=settings.int_distance_ecog,
                                                              int_distance_stn=settings.int_distance_stn)

    dict_ = {
        "int_data": int_data,
        "label_mov": label_mov,
        "act_grid_points": act_grid_points
    }

    out_path_file = os.path.join(settings.out_path_folder, file_name_out) + '.p'
    pickle.dump(dict_, open(out_path_file, "wb"))


#debug:
#f, t, Sxx = signal.spectrogram(x_filtered[7,0,:], sample_rate); plt.pcolormesh(t, f, Sxx); plt.ylim(0,200);

if __name__== "__main__":

    vhdr_filename_paths = Settings.read_all_vhdr_filenames()

    #write_and_interpolate_vhdr(vhdr_filename_paths[31])
    #write_and_interpolate_vhdr(vhdr_filename_paths[0])
    pool = multiprocessing.Pool()
    pool.map(write_and_interpolate_vhdr, vhdr_filename_paths)

