import features
import numpy as np
#import projection
from scipy import sparse
from scipy.sparse.linalg import spsolve
#import cvxpy as cp
from scipy import signal
import preprocessing

def run(gen, seglengths, f_ranges, line_noise, fs, fs_new, filter_fun, num_channels, clip_low=-2, clip_high=2, usemean_=True, normalize=True,\
        normalize_time=30):

    num_features = 8 # later important to be distinguishible for different features
    feature_arr = np.zeros([1, num_channels, num_features])
    
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
        
        # call rereference

        # notch filter and feature estimation
        features_sample = np.zeros([num_channels,num_features])
        for ch_idx, ch in enumerate(np.arange(0,num_channels,1)):
            dat_filt = features.apply_filter(ieeg_batch[ch_idx, :], sample_rate=fs, filter_fun=filter_fun, line_noise=line_noise, seglengths=(fs / seglengths).astype(int))
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
                feature_arr_norm[n_idx,:,:] = np.clip(feature_arr[n_idx,:,:], clip_low, clip_high)
            else:
                if usemean_ is True:   
                    norm_previous = np.mean(feature_arr[n_idx,:,:], axis=0)
                else:
                    norm_previous = np.median(feature_arr[n_idx,:,:], axis=0)
                    
                feature_norm = (feature_arr[cnt_samples,:,:] - norm_previous) / norm_previous
                feature_norm = np.clip(feature_norm, clip_low, clip_high)
                feature_arr_norm = np.concatenate((feature_arr_norm, feature_norm), axis=0) 
            print(cnt_samples)
            cnt_samples += 1 
                