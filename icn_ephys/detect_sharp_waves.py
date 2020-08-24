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
import multiprocessing
from itertools import repeat

BIDS_PATH = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\"
COMB_RUNS_PATH = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\Combined_runs\\"
PATH_OUT = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\SharpWaveAnalysis\\"

class NoValidTroughException(Exception):
    pass

class Waveform_analyzer:

    def __init__(self, sample_rate=1000, line_noise=60, bp_low_cutoff=5, bp_high_cutoff=90):
        self.sample_rate = sample_rate
        self.line_noise = line_noise
        self.filter = mne.filter.create_filter(None, sample_rate, l_freq=bp_low_cutoff, h_freq=bp_high_cutoff,
                                fir_design='firwin', l_trans_bandwidth=5,
                                h_trans_bandwidth=5, filter_length='1000ms')

    def get_peaks_around(self, trough_ind, arr_ind_peaks, filtered_dat):

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

    #def analyze_waveform(self, raw_dat, peak_dist=1, trough_dist=5, label=False, y_contra=None, y_ipsi=None, \
    #                        plot_=False):
    def analyze_waveform(self, ch, dat, subject_id, DETECT_PEAKS):

        label=True;
        plot_=False
        trough_dist=5;

        y_contra = dat[ch]["mov_con"]; # else: y_contra=None;
        y_ipsi = dat[ch]["mov_ips"]# else: y_ipsi=None;
        raw_dat = dat[ch]["data"]

        if DETECT_PEAKS is True:
            peak_dist=5; trough_dist=1;
            raw_dat = -dat[ch]["data"]  # Negative for Peaks
        else:
            peak_dist=1; trough_dist=5;
            raw_dat = dat[ch]["data"]

        # first notch filter data
        dat_notch_filtered = mne.filter.notch_filter(x=raw_dat, Fs=self.sample_rate, trans_bandwidth=7,
            freqs=np.arange(self.line_noise, 4*self.line_noise, self.line_noise),
            fir_design='firwin', verbose=False, notch_widths=1,filter_length=raw_dat.shape[0]-1)

        filtered_dat = scipy.signal.convolve(dat_notch_filtered, self.filter, mode='same')

        peaks = scipy.signal.find_peaks(filtered_dat, distance=peak_dist)[0]
        troughs = scipy.signal.find_peaks(-filtered_dat, distance=trough_dist)[0]

        if plot_ is True:
            plt.figure(figsize=(15,5))
            plt.plot(peaks, filtered_dat[peaks], "xr");
            plt.plot(troughs, filtered_dat[troughs], "ob");
            plt.plot(filtered_dat, color='black'); plt.legend(['peaks', 'trough'])
            plt.show()

        df  = pd.DataFrame()
        sharp_wave = {}
        for trough_idx in troughs:
            try:
                peak_idx_left, peak_idx_right, peak_left, peak_right = self.get_peaks_around(trough_idx,
                                                                                        peaks, filtered_dat)
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
            if (trough_idx - int(5*(1000/self.sample_rate)) <= 0) or \
                (trough_idx + int(5*(1000/self.sample_rate)) >= filtered_dat.shape[0]):
                continue
            # convert 5 ms to sample rate
            sharpness = ((filtered_dat[trough_idx] - filtered_dat[trough_idx-int(5*(1000/self.sample_rate))]) +
                         (filtered_dat[trough_idx] - filtered_dat[trough_idx+int(5*(1000/self.sample_rate))])) / 2

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
                "prominence": np.abs((peak_right + peak_left) / 2 - filtered_dat[trough_idx]), # mV
                "interval" : interval_, # ms
                "decay_time": (peak_idx_left - trough_idx) *(1000/self.sample_rate),
                "rise_time" : (peak_idx_right - trough_idx) *(1000/self.sample_rate),
                "sharpness" : sharpness,
                "rise_steepness" : rise_steepness,
                "decay_steepness" : decay_steepness,
                "slope_ratio" : rise_steepness - decay_steepness,
                "label" : False,
                "MOV_TYPE" : None,
                "y_contra" : None,
                "y_ipsi" : None
            }

            if label is True:
                sharp_wave["label"] = True
                    # movement
                if y_ipsi[trough_idx] > 0:
                    MOV_ = "IPS"
                elif y_contra[trough_idx] > 0:
                    MOV_ = "CON"
                else:
                    MOV_ = "NO_MOV"
                sharp_wave["MOV_TYPE"] = MOV_
                sharp_wave["y_contra"] = y_contra[trough_idx]
                sharp_wave["y_ipsi"] = y_ipsi[trough_idx]
            df = df.append(sharp_wave, ignore_index=True)

        df.to_pickle(PATH_OUT + "sub_"+subject_id+"_ch_"+ch+".p")
        #return df
global dat
def analyze_sharpwaves_subject(subject_id):

    waveform_analyzer = Waveform_analyzer(sample_rate=1000, line_noise=60)

    with open(os.path.join(COMB_RUNS_PATH, "sub_"+subject_id+"_comb.p"), "rb") as handle:
        dat = pickle.load(handle)

    pool = multiprocessing.Pool()

    ch_left = []
    for ch in dat.keys():
        # check if outfile already exists
        if os.path.exists(PATH_OUT + "sub_"+subject_id+"_ch_"+ch+".p") is False:
            ch_left.append(ch)

    print("channel left subject "+str(subject_id))
    print(ch_left)

    pool.starmap(waveform_analyzer.analyze_waveform, zip(ch_left, repeat(dat), repeat(subject_id), repeat(True))) # repeat True for PEAKS

    #for ch in dat.keys():
    #    df = waveform_analyzer.analyze_waveform(dat[ch]["data"], peak_dist=1, trough_dist=12,
    #                     label=True, y_contra=dat[ch]["mov_con"], y_ipsi=dat[ch]["mov_ips"], plot_=False)


    #df.to_pickle(PATH_OUT + "sub_"+subject_id+"_ch_"+ch+".p")

if __name__ == '__main__':

    sub_str = []
    for sub_idx  in np.flip(np.arange(0, 16, 1)):
        print(sub_idx)
        if sub_idx<10:
            subject_id = '00' + str(sub_idx)
        else:
            subject_id = '0' + str(sub_idx)
        sub_str.append(subject_id)
        analyze_sharpwaves_subject(subject_id)
