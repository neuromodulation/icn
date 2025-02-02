import numpy as np
import rereference
import realtime_normalization
import pandas as pd
import time
from numba import jit

def run(gen, features, settings, df_M1):
    """Run "real-time" analysis of neurophysiological data generated by "gen".

    Parameters
    ----------
    gen : generator object
        generator that yields segments of data.
    features: 
        Feature_df object (needs to be initialized beforehand)
    settings : dict
        dictionary of settings such as "seglengths" or "frequencyranges"
    df_M1 : data frame
        data frame with the channel configurations and rereferencing settings.

    Returns
    -------
    features (df) : features defined as in settings in shape [M, N] where N is the time index and 
                    M is the total feature number 
    """

    fs_new = int(settings["resampling_rate"])
    normalize_time = int(settings["normalization_settings"]["normalization_time"])
    offset = int(1000 * settings["bandpass_filter_settings"]["segment_lengths"][0]) # ms
    cnt_samples = 0

    if settings["methods"]["normalization"] is True:
        normalize_samples = int(normalize_time * features.fs)  # normalization is here made for the raw signal
        feature_arr = pd.DataFrame()
    start_time = time.time()
    
    while True:
        ieeg_batch = next(gen, None)
        start_time = time.time()
        if ieeg_batch is None:
            return feature_arr

        # call rereference
        if settings["methods"]["re_referencing"] is True:
            ieeg_batch = rereference.rereference(ieeg_batch, df_M1)

        # now normalize raw data 
        if settings["methods"]["normalization"] is True:
            if cnt_samples == 0:
                raw_arr = ieeg_batch
            else:
                raw_arr = np.concatenate((raw_arr, ieeg_batch), axis=1)
            
            raw_norm = realtime_normalization.realtime_normalization(raw_arr, cnt_samples, normalize_samples, features.fs, \
                    settings["normalization_settings"]["normalization_method"])

            # calculate features
            feature_series = features.estimate_features(raw_norm) 
        else: 
            feature_series = features.estimate_features(ieeg_batch)
        
        if cnt_samples == 0:
            cnt_samples += int(features.fs)
            feature_series["time"] = offset # ms
            feature_arr = pd.DataFrame([feature_series])
            
        else:
            cnt_samples += int(features.fs / fs_new)
            feature_series["time"] = cnt_samples*1000/features.fs # ms
            feature_arr = feature_arr.append(feature_series, ignore_index=True)
        print(f"{str(np.round(cnt_samples / 1000,2))} seconds of data processed.")
        print("took: "+str(round(time.time() - start_time, 2))+" seconds")
            
