import numpy as np 
import pandas as pd
 
def realtime_normalization(raw_arr, cnt_samples, normalize_samples, settings, fs):

    if cnt_samples == 0:
        return raw_arr
    if cnt_samples < normalize_samples:
        n_idx = np.arange(0, cnt_samples, 1)
    else:
        n_idx = np.arange(cnt_samples - normalize_samples, cnt_samples+1, 1)

    if settings["normalization_settings"]["normalization_method"] == "mean":
        norm_previous = np.mean(raw_arr[:, n_idx], axis=1)
    elif settings["normalization_settings"]["normalization_method"] == "median":
        norm_previous = np.median(raw_arr[:, n_idx], axis=1)
    else: 
        raise TypeError("only median and mean is supported as normalization method") 
    raw_norm = (raw_arr[:, -fs:].T - norm_previous) / norm_previous.T

    return raw_norm.T