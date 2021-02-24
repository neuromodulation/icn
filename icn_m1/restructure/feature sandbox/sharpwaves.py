import numpy as np 
import scipy
import mne
import pandas as pd

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

def analyze_waveform(dat, DETECT_PEAKS, sample_rate, bp_low_cutoff=5, bp_high_cutoff=90):
    
    """
    Estimates dataframe with detected sharp wave characetristics.
    Data is bandpass filtered before preprocessing. Data is assumed to be notch filtered beforehand. 

    dat (np array): 1d time vector for used channel
    DETECT_PEAKS (bool): if true PEAKS are analyzed, if false: troughs
    sample_rate (int)
    bp_low_cutoff (int): data is bandpass filtered before preprocessing
    bp_high_cutoff (int): data is bandpass filtered before preprocessing
    """

    if DETECT_PEAKS is True:
        peak_dist=5; trough_dist=1;  # ms distance between detected troughs / peaks 
        raw_dat = -dat  # Negative for Peaks
    else:
        peak_dist=1; trough_dist=5
        raw_dat = dat

    filter_ = mne.filter.create_filter(None, sample_rate, l_freq=bp_low_cutoff, h_freq=bp_high_cutoff,
                                fir_design='firwin', l_trans_bandwidth=5,
                                h_trans_bandwidth=5, filter_length=str(sample_rate)+'ms', verbose=False)

    filtered_dat = scipy.signal.convolve(dat, filter_, mode='same')

    peaks = scipy.signal.find_peaks(filtered_dat, distance=peak_dist)[0]
    troughs = scipy.signal.find_peaks(-filtered_dat, distance=trough_dist)[0]

    df  = pd.DataFrame()
    sharp_wave = {}
    for trough_idx in troughs:
        try:
            peak_idx_left, peak_idx_right, peak_left, peak_right = get_peaks_around(trough_idx,
                                                                    peaks, filtered_dat)
        except NoValidTroughException as e:
            # in this case there are no adjacent two peaks around this trough
            # str(e) could print the exception error message
            #print(str(e))
            continue

        # interval
        if df.shape[0]>0:
            interval_ = (trough_idx - sharp_wave["trough_idx"]) * (1000/sample_rate)
        else:
            # set first interval to zero
            interval_ = 0

        # sharpness calculation, first check if the sharpness can be calculated for that peak or trough
        if (trough_idx - int(5*(1000/sample_rate)) <= 0) or \
            (trough_idx + int(5*(1000/sample_rate)) >= filtered_dat.shape[0]):
            continue

        sharpness = ((filtered_dat[trough_idx] - filtered_dat[trough_idx-int(5*(1000/sample_rate))]) +
                        (filtered_dat[trough_idx] - filtered_dat[trough_idx+int(5*(1000/sample_rate))])) / 2

        # rise_steepness is calculated as the first derivative from trough to peak
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
            "decay_time": (peak_idx_left - trough_idx) *(1000/sample_rate), # ms
            "rise_time" : (peak_idx_right - trough_idx) *(1000/sample_rate), # ms
            "sharpness" : sharpness,
            "rise_steepness" : rise_steepness,
            "decay_steepness" : decay_steepness,
            "slope_ratio" : rise_steepness - decay_steepness
        }

        df = df.append(sharp_wave, ignore_index=True)
    return df 

def get_sharpwave_features(features_, s, fs, data_, ch):
    if s["sharpwave_analysis_settings"]["MaxTroughProminence"] is True or \
        s["sharpwave_analysis_settings"]["MaxTroughSharpness"] is True or \
        s["sharpwave_analysis_settings"]["MeanTroughSharpness"] is True:                  
        
        df_sw = analyze_waveform(data_, DETECT_PEAKS=False, sample_rate=fs, \
            bp_low_cutoff=s["sharpwave_analysis_settings"]["filter_low_cutoff"], \
            bp_high_cutoff=s["sharpwave_analysis_settings"]["filter_high_cutoff"])
        if df_sw.shape[1] == 0:
            return #continue

        if s["sharpwave_analysis_settings"]["MaxTroughProminence"] is True:
            features_['_'.join([ch,'Sharpwave_MaxTroughprominence'])] = df_sw["prominence"].max()
        if s["sharpwave_analysis_settings"]["MaxTroughSharpness"] is True:
            features_['_'.join([ch,'Sharpwave_MaxTroughSharpness'])] = df_sw["sharpness"].max()
        if s["sharpwave_analysis_settings"]["MeanTroughSharpness"] is True:
            features_['_'.join([ch, 'Sharpwave_MeanTroughSharpness'])] = df_sw["sharpness"].mean()
    if s["sharpwave_analysis_settings"]["MaxPeakProminence"] is True or \
        s["sharpwave_analysis_settings"]["MaxPeakSharpness"] is True or \
        s["sharpwave_analysis_settings"]["MeanPeakSharpness"] is True:                  
        df_sw = analyze_waveform(data_, DETECT_PEAKS=True, sample_rate=fs, \
            bp_low_cutoff=s["sharpwave_analysis_settings"]["filter_low_cutoff"], \
            bp_high_cutoff=s["sharpwave_analysis_settings"]["filter_high_cutoff"])
        
        if df_sw.shape[1] == 0:
            return #continue
            
        if s["sharpwave_analysis_settings"]["MaxPeakProminence"] is True:
            features_['_'.join([ch,'Sharpwave_MaxPeakprominence'])] = df_sw["prominence"].max()
        if s["sharpwave_analysis_settings"]["MaxPeakSharpness"] is True:
            features_['_'.join([ch,'Sharpwave_MaxPeakSharpness'])] = df_sw["sharpness"].max()
        if s["sharpwave_analysis_settings"]["MeanPeakSharpness"] is True:
            features_['_'.join([ch, 'Sharpwave_MeanPeakSharpness'])] = df_sw["sharpness"].mean()
        # the sharpness ration between troughs and peaks could be added as well

    return features_