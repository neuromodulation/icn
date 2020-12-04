import filter
import IO
import sys
import numpy as np
import time
import json
import os
import pandas as pd
import mne

def run(gen, settings, df_M1, fs, line_noise, filter_fun, usemean_=True, normalize=True):

    num_features = 8 # later important to be distinguishible for different features
    num_channels = df_M1[(df_M1["used"] == 1) & (df_M1["target"] == 0)].index.shape[0]
    feature_arr = np.zeros([1, num_channels, num_features])
    
    fs_new = int(settings["resamplingrate"])
    normalize_time = int(settings["normalization_time"])
    cnt_samples = 0 
    
    if normalize is True:
        normalize_samples = int(normalize_time*fs_new)
        feature_arr_norm = np.zeros(feature_arr.shape)
    while True:
        ieeg_batch = next(gen, None)
        
        if ieeg_batch is None: 
            if normalize is True:
                return feature_arr_norm
            else: 
                return feature_arr
        
        ### call rereference ###
        #ieeg_batch = reference(ieeg_batch, df_M1)
        
        # notch filter and feature estimation
        features_sample = np.zeros([num_channels,num_features])
        for ch_idx, ch in enumerate(np.arange(0,num_channels,1)):
            dat_filt = filter.apply_filter(ieeg_batch[ch_idx,:], sample_rate=fs, filter_fun=filter_fun, \
                        line_noise=line_noise, seglengths=(fs/np.array(settings["seglengths"])).astype(int))
            features_sample[ch_idx,:] = dat_filt
        
        feature_arr = np.concatenate((feature_arr, np.expand_dims(features_sample, axis=0)), axis=0)
        if normalize is True:
            if cnt_samples < normalize_samples:
                if cnt_samples == 0:
                    n_idx = 0
                else:
                    n_idx = np.arange(0,cnt_samples,1)
            else:
                n_idx = np.arange(cnt_samples-normalize_samples, cnt_samples, 1)

            if cnt_samples == 0:
                feature_arr_norm[n_idx,:,:] = np.clip(feature_arr[n_idx,:,:], settings["clip_low"], \
                                                      settings["clip_high"])
            else:
                if usemean_ is True:   
                    norm_previous = np.mean(feature_arr[n_idx,:,:], axis=0)
                else:
                    norm_previous = np.median(feature_arr[n_idx,:,:], axis=0)
                    
                feature_norm = (feature_arr[cnt_samples,:,:] - norm_previous) / norm_previous
                
                ### Artifact rejection ###
                # feature norm can consist NaN's if norm_previous is zero, therefore cast beforehand
                feature_norm = np.clip(np.nan_to_num(feature_norm), settings["clip_low"], settings["clip_high"])
                feature_arr_norm = np.concatenate((feature_arr_norm, \
                                                   np.expand_dims(feature_norm, axis=0)), axis=0) 
            print(str(np.round(cnt_samples/fs_new,2))+ "s")
            cnt_samples += 1 