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

class Waveform_analyzer:

    def __init__(self, sample_rate=1000, line_noise=60, bp_low_cutoff=5, bp_high_cutoff=90):
        self.sample_rate = sample_rate
        self.line_noise = line_noise
        self.filter = mne.filter.create_filter(None, sample_rate, l_freq=bp_low_cutoff, h_freq=bp_high_cutoff,
                                fir_design='firwin', l_trans_bandwidth=5,
                                h_trans_bandwidth=5, filter_length='1000ms')

    class NoValidTroughException(Exception):
        pass

    def get_peaks_around(trough_ind, arr_ind_peaks, filtered_dat):

        # find all peaks to the right (then left) side, then take the closest one to the trough
        ind_greater = np.where(arr_ind_peaks>trough_ind)[0]
        if ind_greater.shape[0] == 0:
            raise NoValidTroughException("No valid trough")
        val_ind_greater = arr_ind_peaks[ind_greater]
        peak_right_idx = arr_ind_peaks[ind_greater[np.argsort(val_ind_greater)[0]]]

        ind_smaller = np.where(arr_ind_peaks<trough_ind)[0]
        if ind_smaller.shape[0] == 0:
            raise NoValidTroughException("No valid trough")

        val_ind_smaller = arr_ind_peaks[ind_smaller]
        peak_left_idx = arr_ind_peaks[ind_smaller[np.argsort(val_ind_smaller)[-1]]]

        return peak_left_idx, peak_right_idx, filtered_dat[peak_left_idx], filtered_dat[peak_right_idx]

    def analyze_waveform(raw_dat, peak_dist=1, trough_dist=5, mov_con=False, mov_ips=False):

        filtered_dat = scipy.signal.convolve(raw_dat, self.filter, mode='same')
        peaks = scipy.signal.find_peaks(filtered_dat, distance=peak_dist)[0]
        troughs = scipy.signal.find_peaks(-filtered_dat, distance=trough_dist)[0]

        df  = pd.DataFrame()
        sharp_wave = {}
        for trough_idx in troughs:
            try:
                peak_idx_left, peak_idx_right, peak_left, peak_right = get_peaks_around(trough_idx, peaks, filtered_dat)
            except NoValidTroughException as e:
                # in this case there is no adjacent two peaks around this trough
                print(str(e))
                continue

            # interval
            if df.shape[0]>0:
                interval_ = (trough_idx - sharp_wave["trough_idx"]) * (1000/self.sample_rate)
            else:
                # set first interval to zero
                interval_ = 0

            # sharpness
            if (trough_idx - int(5*(1000/self.sample_rate)) < 0) or \
                (trough_idx + int(5*(1000/self.sample_rate)) > filtered_dat.shape[0]):
                continue
            # convert 5 ms to sample rate
            sharpness = (filtered_dat[trough_idx-int(5*(1000/self.sample_rate))] +
                         filtered_dat[trough_idx+int(5*(1000/self.sample_rate))]) / 2

            # rise_steepness, first der. from trough to peak
            # here  + 1 due to python syntax, s.t. the last element is included
            rise_steepness = np.max(np.diff(filtered_dat[peak_idx_left : trough_idx+1]))

            # decay_steepness
            decay_steepness = np.max(np.diff(filtered_dat[trough_idx : peak_idx_right+1]))

            sharp_wave  = {
                "peak_left" : peak_left,
                "peak_right" : peak_right,
                "peak_idx_left" : peak_idx_left,
                "peak_idx_right" : peak_idx_right,
                "trough" : filtered_dat[trough_idx], # mV
                "trough_idx" : trough_idx,
                "width" : peak_idx_right - peak_idx_left, # ms
                "prominence": np.abs((peak_right + peak_left) / 2 - trough_idx), # mV
                "interval" : interval_, # ms
                "decay_time": (peak_idx_left - trough_idx) *(1000/self.sample_rate),
                "rise_time" : (peak_idx_right - trough_idx) *(1000/self.sample_rate),
                "sharpness" : sharpness,
                "rise_steepness" : rise_steepness,
                "decay_steepness" : decay_steepness,
                "slope_ratio" : rise_steepness - decay_steepness
            }
            df = df.append(sharp_wave, ignore_index=True)
        return df 
