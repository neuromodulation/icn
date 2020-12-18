import numpy as np

import mne

import filter
from preprocessing import rereference

def run(gen, settings, df_M1, fs, line_noise, filter_fun, usemean_=True, normalize=True, include_label=False, method='bandpass'):
    num_features = len(settings['frequencyranges'])  # later important to be distinguishible for different features
    num_channels = df_M1[(df_M1["used"] == 1) & (df_M1["target"] == 0)].index.shape[0]
    feature_arr = np.zeros([1, num_channels, num_features])
    fs_new = int(settings["resamplingrate"])
    normalize_time = int(settings["normalization_time"])
    cnt_samples = 0

    if include_label == True:
        label = np.zeros([1])

    if normalize == True:
        normalize_samples = int(normalize_time * fs_new)
        feature_arr_norm = np.zeros(feature_arr.shape)
    while True:
        ieeg_batch = next(gen, None)

        if ieeg_batch == None:
            if normalize == True:
                if include_label == True:
                    return feature_arr_norm, label
                else:
                    return feature_arr_norm
            else:
                if include_label == True:
                    return feature_arr, label
                else:
                    return feature_arr

        ### call rereference ###
        ieeg_batch = rereference(ieeg_batch, df_M1)

        # feature estimation by bandpass filtering
        if method == 'bandpass':
            features_sample = np.zeros([num_channels, num_features])
            for ch_idx, ch in enumerate(np.arange(0, num_channels, 1)):
                dat_filt = filter.apply_filter(ieeg_batch[ch_idx, :], sample_rate=fs, filter_fun=filter_fun, \
                                               line_noise=line_noise,
                                               seglengths=(fs / np.array(settings["seglengths"])).astype(int))
                features_sample[ch_idx, :] = dat_filt

        # feature estimation by wavelet transform
        if method == 'wavelet1':
            batch_notch = mne.filter.notch_filter(x=ieeg_batch, Fs=fs, trans_bandwidth=7, filter_length='auto',
                                                  freqs=np.arange(line_noise, 4 * line_noise + 1, line_noise),
                                                  fir_design='firwin', notch_widths=1, verbose=False)
            freq_max = max(max(settings['frequencyranges'], key=lambda x: x[1]))
            freq_min = min(min(settings['frequencyranges'], key=lambda x: x[0]))
            freqs = np.arange(freq_min, freq_max, 2)
            n_cycles = log2(freqs)
            features_sample = np.zeros([num_channels, num_features])
            reshape_batch = np.reshape(batch_notch, (1, batch_notch.shape[0], batch_notch.shape[1]))
            dat_tf = mne.time_frequency.tfr_array_morlet(epoch_data=reshape_batch, sfreq=fs, freqs=freqs, n_cycles=n_cycles,
                                      output='avg_power')
            for idx, f_range in enumerate(settings['frequencyranges']):
                features_sample[:, idx] = np.mean(np.mean(dat_tf[:, (f_range[0] - freq_min)//2:f_range[0]//2, :], axis=1)[:,
                                                  -fs // settings['seglengths'][idx]:], axis=1)
        # feature estimation by wavelet transform
        if method == 'wavelet2':
            data = mne.filter.notch_filter(x=ieeg_batch, Fs=fs, trans_bandwidth=7, filter_length='auto',
                                                  freqs=np.arange(line_noise, 4 * line_noise + 1, line_noise),
                                                  fir_design='firwin', notch_widths=1, verbose=False)
            features_sample = np.zeros([num_channels, num_features])
            for idx, f_range in enumerate(settings['frequencyranges']):
                freqs = np.arange(f_range[0], f_range[1], 1)
                n_cycles = 1
                batch_notch = data[:, int(-1 / settings['seglengths'][idx] * data.shape[1]):]
                reshape_batch = np.reshape(batch_notch, (1, batch_notch.shape[0], batch_notch.shape[1]))
                dat_tf = mne.time_frequency.tfr_array_morlet(epoch_data=reshape_batch, sfreq=fs, freqs=freqs,
                                                             n_cycles=n_cycles,
                                                             output='avg_power')
                features_sample[:,idx] = np.mean(dat_tf, axis=(1,2))

        if cnt_samples == 0:
            feature_arr = np.expand_dims(features_sample, axis=0)
            if include_label == True:
                label = ieeg_batch[-1, -1]
        else:
            feature_arr = np.concatenate((feature_arr, np.expand_dims(features_sample, axis=0)), axis=0)
            if include_label == True:
                label = np.vstack((label, ieeg_batch[-1, -1]))

        if normalize is True:
            if cnt_samples < normalize_samples:
                if cnt_samples == 0:
                    n_idx = 0
                else:
                    n_idx = np.arange(0, cnt_samples, 1)
            else:
                n_idx = np.arange(cnt_samples - normalize_samples, cnt_samples+1, 1)

            if cnt_samples == 0:
                feature_arr_norm[n_idx, :, :] = np.clip(feature_arr[n_idx, :, :], settings["clip_low"], \
                                                        settings["clip_high"])
            else:
                if usemean_ is True:
                    norm_previous = np.mean(feature_arr[n_idx, :, :], axis=0)
                else:
                    norm_previous = np.median(feature_arr[n_idx, :, :], axis=0)

                feature_norm = (feature_arr[cnt_samples, :, :] - norm_previous) / norm_previous

                ### Artifact rejection ###
                feature_norm = np.clip(feature_norm, settings["clip_low"], settings["clip_high"])
                feature_arr_norm = np.concatenate((feature_arr_norm, \
                                                   np.expand_dims(feature_norm, axis=0)), axis=0)
        #print(str(np.round(cnt_samples/fs_new,2))+ "s")
        cnt_samples += 1