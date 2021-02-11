import mne
import numpy as np
import scipy
from matplotlib import pyplot as plt
import kalmanfilter
import filter
import sharpwaves
import pandas as pd
from multiprocessing import Process, Manager

class Features:
    
    def __init__(self, s, fs, line_noise, channels):
        """
        s (dict) : json settings
        """

        self.ch_names = channels
        self.s = s # settings
        self.fs = fs
        self.line_noise = line_noise
        self.seglengths = (self.fs / np.array(s["bandpass_filter_settings"]["segment_lengths"])).astype(int)
        self.KF_dict = {}
        
        if s["methods"]["bandpass_filter"] is True:
            self.filter_fun = filter.calc_band_filters(s["bandpass_filter_settings"]["frequency_ranges"], \
                                                sample_rate=fs, filter_len=fs + 1)
        
        for channel in self.ch_names:
            if s["methods"]["kalman_filter"] is True:
                for bp_feature in s["bandpass_filter_settings"]["bandpower_features"]:
                    if s["bandpass_filter_settings"]["bandpower_features"][bp_feature] is False:
                        continue
                    for fband_idx, f_band in enumerate(s["bandpass_filter_settings"]["feature_labels"]): 
                        if s["kalman_filter_settings"]["frequency_bands"][fband_idx] is True: 
                            self.KF_dict[channel+"_"+bp_feature+"_"+f_band] = \
                                kalmanfilter.define_KF(s["kalman_filter_settings"]["Tp"], \
                                            s["kalman_filter_settings"]["sigma_w"], \
                                            s["kalman_filter_settings"]["sigma_v"])
        
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
        
        manager = Manager()
        features_ = manager.dict() #features_ = {}

        
        # notch filter data before feature estimation 
        data = mne.filter.notch_filter(x=data, Fs=self.fs, trans_bandwidth=7,
            freqs=np.arange(self.line_noise, 4*self.line_noise, self.line_noise),
            fir_design='firwin', verbose=False, notch_widths=3, filter_length=data.shape[1]-1)
        
        if self.s["methods"]["bandpass_filter"] is True:
            dat_filtered = filter.apply_filter(data, self.filter_fun) # shape (bands, time)
        else:
            dat_filtered = None
        
        self.data = data
        self.dat_filtered = dat_filtered

        job = [Process(target=self.est_ch, args=(features_, ch_idx, ch)) for ch_idx, ch in enumerate(self.ch_names)]
        _ = [p.start() for p in job]
        _ = [p.join() for p in job]

        return dict(features_)

                    
    def est_ch(self, features_, ch_idx, ch):
            
        for f_band in range(len(self.s["bandpass_filter_settings"]["frequency_ranges"])):
            seglength = min((self.seglengths[f_band], self.dat_filtered.shape[-1]))
            for bp_feature in self.s["bandpass_filter_settings"]["bandpower_features"]:
                if bp_feature == "activity" and self.s["bandpass_filter_settings"]["bandpower_features"][bp_feature] is True:
                    feature_calc = np.var(self.dat_filtered[ch_idx, f_band, -seglength:])                                
                elif bp_feature == "mobility" and self.s["bandpass_filter_settings"]["bandpower_features"][bp_feature] is True:
                    deriv_variance = np.var(np.diff(self.dat_filtered[ch_idx, f_band, -seglength:]))
                    feature_calc = np.sqrt(deriv_variance / np.var(self.dat_filtered[ch_idx, f_band, -seglength:]))
                elif bp_feature == "complexity" and self.s["bandpass_filter_settings"]["bandpower_features"][bp_feature] is True:
                    dat_deriv = np.diff(self.dat_filtered[ch_idx, f_band, -seglength:])
                    deriv_variance = np.var(dat_deriv)
                    mobility = np.sqrt(deriv_variance / np.var(self.dat_filtered[ch_idx, f_band, -seglength:]))
                    dat_deriv_2 = np.diff(dat_deriv)
                    dat_deriv_2_var = np.var(dat_deriv_2)
                    deriv_mobility = np.sqrt(dat_deriv_2_var / deriv_variance)
                    feature_calc = deriv_mobility / mobility
                if self.s["kalman_filter_settings"]["frequency_bands"][f_band] is True:
                    self.KF_dict[ch+"_"+bp_feature+"_"+self.s["bandpass_filter_settings"]["feature_labels"][f_band]].predict()
                    self.KF_dict[ch+"_"+bp_feature+"_"+self.s["bandpass_filter_settings"]["feature_labels"][f_band]].update(feature_calc)
                    feature_calc = self.KF_dict[ch+"_"+bp_feature+"_"+self.s["bandpass_filter_settings"]["feature_labels"][f_band]].x[0] # filtered sensor signal
                
                feature_name = ch+"_bandpass_"+bp_feature+"_"+self.s["bandpass_filter_settings"]["feature_labels"][f_band]
                features_[feature_name] = feature_calc
        
        if self.s["methods"]["raw_hjorth"] is True: 
            feature_name = ch+"_RawHjorth_Activity"
            features_[ch+"_RawHjorth_Activity"] = np.var(self.data[ch_idx,:])
            deriv_variance = np.var(np.diff(self.data[ch_idx,:]))
            mobility = np.sqrt(deriv_variance / np.var(self.data[ch_idx,:]))
            features_[ch+"_RawHjorth_Mobility"] = mobility
            
            dat_deriv_2_var = np.var(np.diff(np.diff(self.data[ch_idx,:])))
            deriv_mobility = np.sqrt(dat_deriv_2_var / np.var(np.diff(self.data[ch_idx,:])))
            features_[ch+"_RawHjorth_Complexity"] = deriv_mobility / mobility
        
        if self.s["methods"]["return_raw"] is True:
            features_[ch+"_raw"] = self.data[ch_idx, -1] # this basically just subsamles raw data
        
        # check if any sharpwave feature was selected in settings
        if self.s["methods"]["sharpwave_analysis"] is True: 
            if self.s["sharpwave_analysis_settings"]["MaxTroughProminence"] is True or \
                self.s["sharpwave_analysis_settings"]["MaxTroughSharpness"] is True or \
                self.s["sharpwave_analysis_settings"]["MeanTroughSharpness"] is True:                  
                
                df_sw = sharpwaves.analyze_waveform(self.data[ch_idx,:], DETECT_PEAKS=False, sample_rate=self.fs, \
                    bp_low_cutoff=self.s["sharpwave_analysis_settings"]["filter_low_cutoff"], \
                    bp_high_cutoff=self.s["sharpwave_analysis_settings"]["filter_high_cutoff"])
                if df_sw.shape[1] == 0:
                    return #continue

                if self.s["sharpwave_analysis_settings"]["MaxTroughProminence"] is True:
                    features_[ch+"_Sharpwave_MaxTroughProminence"] = df_sw["prominence"].max()
                if self.s["sharpwave_analysis_settings"]["MaxTroughSharpness"] is True:
                    features_[ch+"_Sharpwave_MaxTroughSharpness"] = df_sw["sharpness"].max()
                if self.s["sharpwave_analysis_settings"]["MeanTroughSharpness"] is True:
                    features_[ch+"_Sharpwave_MeanTroughSharpness"] = df_sw["sharpness"].mean()
            if self.s["sharpwave_analysis_settings"]["MaxPeakProminence"] is True or \
                self.s["sharpwave_analysis_settings"]["MaxPeakSharpness"] is True or \
                self.s["sharpwave_analysis_settings"]["MeanPeakSharpness"] is True:                  
                df_sw = sharpwaves.analyze_waveform(self.data[ch_idx,:], DETECT_PEAKS=True, sample_rate=self.fs, \
                    bp_low_cutoff=self.s["sharpwave_analysis_settings"]["filter_low_cutoff"], \
                    bp_high_cutoff=self.s["sharpwave_analysis_settings"]["filter_high_cutoff"])
                
                if df_sw.shape[1] == 0:
                    return #continue
                    
                if self.s["sharpwave_analysis_settings"]["MaxPeakProminence"] is True:
                    features_[ch+"_Sharpwave_MaxPeakprominence"] = df_sw["prominence"].max()
                if self.s["sharpwave_analysis_settings"]["MaxPeakSharpness"] is True:
                    features_[ch+"_Sharpwave_MaxPeakSharpness"] = df_sw["sharpness"].max()
                if self.s["sharpwave_analysis_settings"]["MeanPeakSharpness"] is True:
                    features_[ch+"_Sharpwave_MeanPeakSharpness"] = df_sw["sharpness"].mean()
                # the sharpness ration between troughs and peaks could be added as well
