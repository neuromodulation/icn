import mne
import numpy as np
import scipy


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


def apply_filter(dat_, filter_fun):
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


def get_features(ieeg_batch, settings, methods, fs, filter_fun=None):
    """Extract features selected via methods and return as array.

    Parameters
    ----------
    ieeg_batch : array
        data of shape (n_channels, n_samples)
    settings : dict
        dictionary of settings such as "seglengths" or "frequencyranges"
    methods : str or iterable (list, tuple) of str
        features to be extracted. Includes "bandpass", "mobility", "complexity", "wavelet1"
    fs : float/int
        sampling frequency of input data
    filter_fun : array, optional
        output of calc_band_filters (filter (bandpass) used with scipy.signal.convolve)

    Returns
    -------
    features_sample : array
        features array of shape (n_channels, n_features)
    """

    features_list = []
    # feature estimation by bandpass filtering
    if any(name in ('bandpass', 'mobility', 'complexity') for name in methods):
        dat_filtered = apply_filter(ieeg_batch, filter_fun)
        seglengths = (fs / np.array(settings["seglengths"])).astype(int)

        dat_variance = np.zeros((dat_filtered.shape[0], dat_filtered.shape[1]))
        for filt in range(filter_fun.shape[0]):
            seglength = min((seglengths[filt], dat_filtered.shape[-1]))
            for chan in range(dat_filtered.shape[0]):
                dat_variance[chan, filt] = np.var(dat_filtered[chan, filt, -seglength:])
        if 'bandpass' in methods:
            features_list.append(dat_variance)
        if any(name in ('mobility', 'complexity') for name in methods):
            dat_deriv = np.diff(dat_filtered)
            deriv_variance = np.zeros((dat_filtered.shape[0], dat_filtered.shape[1]))
            dat_mobility = np.zeros((dat_filtered.shape[0], dat_filtered.shape[1]))
            for filt in range(filter_fun.shape[0]):
                seglength = min((seglengths[filt], dat_deriv.shape[-1]))
                for chan in range(dat_filtered.shape[0]):
                    deriv_variance[chan, filt] = np.var(dat_deriv[chan, filt, -seglength:])
                    dat_mobility[chan, filt] = np.sqrt(deriv_variance[chan, filt] / dat_variance[chan, filt])
            if 'mobility' in methods:
                features_list.append(dat_mobility)
            if 'complexity' in methods:
                dat_deriv_2 = np.diff(dat_deriv)
                deriv_2_variance = np.zeros((dat_filtered.shape[0], dat_filtered.shape[1]))
                deriv_mobility = np.zeros((dat_filtered.shape[0], dat_filtered.shape[1]))
                dat_complexity = np.zeros((dat_filtered.shape[0], dat_filtered.shape[1]))
                for filt in range(filter_fun.shape[0]):
                    seglength = min((seglengths[filt], dat_deriv_2.shape[-1]))
                    for chan in range(dat_filtered.shape[0]):
                        deriv_2_variance[chan, filt] = np.var(dat_deriv_2[chan, filt, -seglength:])
                        deriv_mobility[chan, filt] = np.sqrt(deriv_2_variance[chan, filt] / deriv_variance[chan, filt])
                        dat_complexity[chan, filt] = deriv_mobility[chan, filt] / dat_mobility[chan, filt]
                features_list.append(dat_complexity)

    # feature estimation by wavelet transform
    if 'wavelet ' in methods:
        seglengths = (fs / np.array(settings["seglengths"])).astype(int)
        freq_max = max(max(settings['frequencyranges'], key=lambda x: x[1]))
        freq_min = min(min(settings['frequencyranges'], key=lambda x: x[0]))
        freqs = np.arange(freq_min, freq_max, 2)
        n_cycles = np.log2(freqs)
        reshape_batch = np.reshape(ieeg_batch, (1, ieeg_batch.shape[0], ieeg_batch.shape[1]))
        dat_tf = mne.time_frequency.tfr_array_morlet(epoch_data=reshape_batch, sfreq=fs, freqs=freqs,
                                                     n_cycles=n_cycles, output='avg_power')
        dat_wavelet = np.zeros([ieeg_batch.shape[0], len(settings['frequencyranges'])])
        for idx, f_range in enumerate(settings['frequencyranges']):
            dat_wavelet[:, idx] = np.mean(np.mean(dat_tf[:, (f_range[0] - freq_min) // 2:f_range[0] // 2, :],
                                                  axis=1)[:, -seglengths[idx]:], axis=1)
        features_list.append(dat_wavelet)

    # feature estimation by wavelet transform
    if 'wavelet2' in methods:
        seglengths = (fs / np.array(settings["seglengths"])).astype(int)
        dat_wavelet2 = np.zeros([ieeg_batch.shape[0], len(seglengths)])
        for idx, f_range in enumerate(settings['frequencyranges']):
            data = ieeg_batch[:, -seglengths[idx]:]
            freqs = np.arange(f_range[0], f_range[1], 1)
            n_cycles = 1
            reshape_batch = np.reshape(data, (1, data.shape[0], data.shape[1]))
            dat_tf = mne.time_frequency.tfr_array_morlet(epoch_data=reshape_batch, sfreq=fs, freqs=freqs,
                                                         n_cycles=n_cycles, output='avg_power')
            dat_wavelet2[:, idx] = np.mean(dat_tf, axis=(1, 2))
        features_list.append(dat_wavelet2)

    features_sample = np.hstack(features_list)
    return features_sample
