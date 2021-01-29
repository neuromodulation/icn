import mne
import numpy as np
import scipy
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter
import pandas as pd

class NoValidTroughException(Exception):
    pass

class features:
    
    def __init__(self, s, fs, line_noise, channels):
        """
        s (dict) : json settings
        """
        
        def define_KF(Tp, sigma_w, sigma_v):
            """
            define Kalman Filter according to White Noise Acceleratin model
            """

            f = KalmanFilter (dim_x=2, dim_z=1)
            f.x = np.array([0, 1])# x here sensor signal and it's first derivative
            f.F = np.array([[1, Tp], [0, 1]])
            f.H = np.array([[1, 0]])
            f.R = sigma_v
            f.Q = np.array([[(sigma_w**2)*(Tp**3)/3, (sigma_w**2)*(Tp**2)/2],\
                            [(sigma_w**2)*(Tp**2)/2, (sigma_w**2)*Tp]])
            f.P = np.cov([[1, 0], [0, 1]]) 
            return f
        
        def calc_band_filters(f_ranges, sample_rate, filter_len="1000ms", l_trans_bandwidth=4, h_trans_bandwidth=4):
            """"Calculate bandpass filters with adjustable length for given frequency ranges.

            This function returns for the given frequency band ranges the filter coefficients with length "filter_len".
            Thus the filters can be sequentially used for band power estimation.

            Parameters
            ----------
            f_ranges : list of lists
                frequency ranges.
            sample_rate : float
                sampling frequency.
            filter_len : str, optional
                length of the filter. Human readable (e.g."1000ms" or "1s"). Default is "1000ms"
            l_trans_bandwidth : int/float, optional
                Length of the lower transition band. The default is 4.
            h_trans_bandwidth : int/float, optional
                Length of the higher transition band. The default is 4.

            Returns
            -------
            filter_fun : array
                filter coefficients stored in array of shape (n_franges, filter_len (in samples))
            """
            filter_list = []
            for a, f_range in enumerate(f_ranges):
                h = mne.filter.create_filter(None, sample_rate, l_freq=f_range[0], h_freq=f_range[1], fir_design='firwin',
                                             l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth,
                                             filter_length=filter_len)
                filter_list.append(h)
            filter_fun = np.vstack(filter_list)
            return filter_fun
        
        self.ch_names = channels
        self.s = s # settings
        self.fs = fs
        self.line_noise = line_noise
        self.seglengths = (self.fs / np.array(s["Seglengths"])).astype(int)
        self.KF_dict = {}
        
        if s["BandpowerMethod"]["BandpassFilter"] is True:
            self.filter_fun = calc_band_filters(s['FrequencyRanges'], \
                                                sample_rate=fs, filter_len=fs + 1)
        
        for channel in self.ch_names:
            if np.sum(s["KalmanfilterFrequencybands"]) > 0:
                for bp_method in s["BandpowerMethod"]:
                    for bp_feature in s["BandpowerFeatures"]:
                        for fband_idx, f_band in enumerate(s["FeatureLabels"]): 
                            if s["KalmanfilterFrequencybands"][fband_idx] is True: 
                                self.KF_dict[channel+"_"+bp_method+"_"+bp_feature+"_"+f_band] = \
                                    define_KF(s["KalmanfilterParams"]["Tp"], \
                                             s["KalmanfilterParams"]["sigma_w"], \
                                             s["KalmanfilterParams"]["sigma_v"])
                                
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
    
    def apply_filter(self, dat_, filter_fun):
        """Apply previously calculated (bandpass) filters to data.

        Parameters
        ----------
        dat_ : array (n_samples, ) or (n_channels, n_samples)
            segment of data.
        filter_fun : array
            output of calc_band_filters.

        Returns
        -------
        filtered : array
            (n_chan, n_fbands, filter_len) array conatining the filtered signal
            at each freq band, where n_fbands is the number of filter bands used to decompose the signal
        """    
        filter_len = max(filter_fun.shape[1], dat_.shape[-1])
        if dat_.ndim == 1:
            filtered = np.zeros((1, filter_fun.shape[0], filter_len))
            for filt in range(filter_fun.shape[0]):
                filtered[0, filt, :] = scipy.signal.convolve(filter_fun[filt, :], dat_, mode='same')
        elif dat_.ndim == 2:
            filtered = np.zeros((dat_.shape[0], filter_fun.shape[0], filter_len))
            for chan in range(dat_.shape[0]):
                for filt in range(filter_fun.shape[0]):
                    filtered[chan, filt, :] = scipy.signal.convolve(filter_fun[filt, :], dat_[chan, :], mode='same')
        else:
            raise ValueError('dat_ needs to have either 1 or 2 dimensions. dat_ had {0} dimensions'.format(dat_.ndim))

        return filtered
    
    def analyze_waveform(self, dat, DETECT_PEAKS, sample_rate, line_noise, bp_low_cutoff=5, bp_high_cutoff=90):
    
        """
        Estimates dataframe with detected sharp wave characetristics.
        Data is bandpass filtered before preprocessing. Data is assumed to be notch filtered beforehand. 

        dat (np array): 1d time vector for used channel
        DETECT_PEAKS (bool): if true PEAKS are analyzed, if false: troughs
        sample_rate (int)
        line_noise (int)
        bp_low_cutoff (int): data is bandpass filtered before preprocessing
        bp_high_cutoff (int): data is bandpass filtered before preprocessing
        """

        if DETECT_PEAKS is True:
            peak_dist=5; trough_dist=1;  # ms distance between detected troughs / peaks 
            raw_dat = -dat  # Negative for Peaks
        else:
            peak_dist=1; trough_dist=5;
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
                peak_idx_left, peak_idx_right, peak_left, peak_right = self.get_peaks_around(trough_idx,
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
        
    def estimate_features(self, data):
        """
        
        Calculate features, as defined in settings.json
        Features are based on bandpower, raw Hjorth parameters and sharp wave characteristics. 
        
        data (np array) : (channels, time)
        
        returns: 
        dat (pd Dataframe) with naming convention: channel_method_feature_(f_band)
        """
        # this is done in a lot of loops unfortunately, 
        # what could be done is to extract the outer channel loop, which could run in parallel 
        
        features_ = {}
        
        # notch filter data before feature estimation 
        data = mne.filter.notch_filter(x=data, Fs=self.fs, trans_bandwidth=7,
            freqs=np.arange(self.line_noise, 4*self.line_noise, self.line_noise),
            fir_design='firwin', verbose=False, notch_widths=1, filter_length=data.shape[1]-1)
        
        for ch_idx, ch in enumerate(self.ch_names):
            for bp_method in self.s["BandpowerMethod"]:
                if bp_method == "BandpassFilter" and self.s["BandpowerMethod"][bp_method] is True:
                    dat_filtered = self.apply_filter(data, self.filter_fun) # shape (bands, time)
                elif bp_method == "Wavelet" and self.s["BandpowerMethod"][bp_method] is True:
                    freq_max = max(max(self.s["FrequencyRanges"], key=lambda x: x[1]))
                    freq_min = min(min(self.s["FrequencyRanges"], key=lambda x: x[0]))
                    freqs = np.arange(freq_min, freq_max, 2)
                    reshape_batch = np.reshape(data, (1, data.shape[0], data.shape[1]))
                    dat_tf = mne.time_frequency.tfr_array_morlet(epoch_data=reshape_batch, sfreq=self.fs, \
                                                    freqs=freqs, n_cycles=5, output='avg_power')
                    dat_filtered = dat_tf # this is wrong, needs to be redone
                else:
                    continue
                for f_band in range(len(self.s["FrequencyRanges"])):
                    seglength = min((self.seglengths[f_band], dat_filtered.shape[-1]))
                    for bp_feature in self.s["BandpowerFeatures"]:
                        if bp_feature == "Activity" and self.s["BandpowerFeatures"][bp_feature] is True:
                            feature_calc = np.var(dat_filtered[ch_idx, f_band, -seglength:])                                
                        elif bp_feature == "Mobility" and self.s["BandpowerFeatures"][bp_feature] is True:
                            deriv_variance = np.var(np.diff(dat_filtered[ch_idx, f_band, -seglength:]))
                            feature_calc = np.sqrt(deriv_variance / np.var(dat_filtered[ch_idx, f_band, -seglength:]))
                        elif bp_feature == "Complexity" and self.s["BandpowerFeatures"][bp_feature] is True:
                            dat_deriv = np.diff(dat_filtered[ch_idx, f_band, -seglength:])
                            deriv_variance = np.var(dat_deriv)
                            mobility = np.sqrt(deriv_variance / np.var(dat_filtered[ch_idx, f_band, -seglength:]))
                            dat_deriv_2 = np.diff(dat_deriv)
                            dat_deriv_2_var = np.var(dat_deriv_2)
                            deriv_mobility = np.sqrt(dat_deriv_2_var / deriv_variance)
                            feature_calc = deriv_mobility / mobility
                        if self.s["KalmanfilterFrequencybands"][f_band] is True:
                            self.KF_dict[ch+"_"+bp_method+"_"+bp_feature+"_"+self.s["FeatureLabels"][f_band]].predict()
                            self.KF_dict[ch+"_"+bp_method+"_"+bp_feature+"_"+self.s["FeatureLabels"][f_band]].update(feature_calc)
                            feature_calc = self.KF_dict[ch+"_"+bp_method+"_"+bp_feature+"_"+self.s["FeatureLabels"][f_band]].x[0] # filtered sensor signal
                        
                        feature_name = ch+"_"+bp_method+"_"+bp_feature+"_"+self.s["FeatureLabels"][f_band]
                        features_[feature_name] = feature_calc
            
            if self.s["RawHjorthParams"] is True: 
                feature_name = ch+"_RawHjorth_Activity"
                features_[ch+"_RawHjorth_Activity"] = np.var(data[ch_idx,:])
                deriv_variance = np.var(np.diff(data[ch_idx,:]))
                mobility = np.sqrt(deriv_variance / np.var(data[ch_idx,:]))
                features_[ch+"_RawHjorth_Mobility"] = mobility
                
                dat_deriv_2_var = np.var(np.diff(np.diff(data[ch_idx,:])))
                deriv_mobility = np.sqrt(dat_deriv_2_var / np.var(np.diff(data[ch_idx,:])))
                features_[ch+"_RawHjorth_Complexity"] = deriv_mobility / mobility
            
            # check if any sharpwave feature was selected in settings
            if sum(self.s["Sharpwave"].values()) > 0: 
                if self.s["Sharpwave"]["MaxTroughProminence"] is True or \
                    self.s["Sharpwave"]["MaxTroughSharpness"] is True or \
                    self.s["Sharpwave"]["MeanTroughSharpness"] is True:                  
                    
                    df_sw = self.analyze_waveform(data[ch_idx,:], DETECT_PEAKS=False, sample_rate=self.fs, \
                        line_noise=self.line_noise, bp_low_cutoff=self.s["SharpwaveFilter"]["bp_low_cutoff"], \
                                                bp_high_cutoff=self.s["SharpwaveFilter"]["bp_high_cutoff"])
                    if df_sw.shape[1] == 0:
                        continue
                    if self.s["Sharpwave"]["MaxTroughProminence"] is True:
                        features_[ch+"_Sharpwave_MaxTroughProminence"] = df_sw["prominence"].max()
                    if self.s["Sharpwave"]["MaxTroughSharpness"] is True:
                        features_[ch+"_Sharpwave_MaxTroughSharpness"] = df_sw["sharpness"].max()
                    if self.s["Sharpwave"]["MeanTroughSharpness"] is True:
                        features_[ch+"_Sharpwave_MeanTroughSharpness"] = df_sw["sharpness"].mean()
                if self.s["Sharpwave"]["MaxPeakProminence"] is True or \
                    self.s["Sharpwave"]["MaxPeakSharpness"] is True or \
                    self.s["Sharpwave"]["MeanPeakSharpness"] is True:                  
                    df_sw = self.analyze_waveform(data[ch_idx,:], DETECT_PEAKS=True, sample_rate=self.fs, \
                        line_noise=self.line_noise, bp_low_cutoff=self.s["SharpwaveFilter"]["bp_low_cutoff"], \
                                                bp_high_cutoff=self.s["SharpwaveFilter"]["bp_high_cutoff"])
                    
                    if df_sw.shape[1] == 0:
                        continue
                    if self.s["Sharpwave"]["MaxPeakProminence"] is True:
                        features_[ch+"_Sharpwave_MaxPeakprominence"] = df_sw["prominence"].max()
                    if self.s["Sharpwave"]["MaxPeakSharpness"] is True:
                        features_[ch+"_Sharpwave_MaxPeakSharpness"] = df_sw["sharpness"].max()
                    if self.s["Sharpwave"]["MeanPeakSharpness"] is True:
                        features_[ch+"_Sharpwave_MeanPeakSharpness"] = df_sw["sharpness"].mean()
                    # the sharpness ration between troughs and peaks could be added as well
    
        return features_

                    