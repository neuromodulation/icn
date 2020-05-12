import numpy as np 
import mne 
import scipy

def calc_band_filters(f_ranges, sample_rate, filter_len=1001, l_trans_bandwidth=4, h_trans_bandwidth=4):
    """
    This function returns for the given frequency band ranges filter coefficients with with length "filter_len"
    Thus the filters can be sequentially used for band power estimation
    """
    filter_fun = np.zeros([len(f_ranges), filter_len])

    for a, f_range in enumerate(f_ranges):
        h = mne.filter.create_filter(None, sample_rate, l_freq=f_range[0], h_freq=f_range[1], 
                            fir_design='firwin', l_trans_bandwidth=l_trans_bandwidth, 
                            h_trans_bandwidth=h_trans_bandwidth, filter_length='1000ms')

        filter_fun[a, :] = h
    return filter_fun

def apply_filter(dat_, sample_rate, filter_fun, line_noise, seglengths):
    """
    For a given channel, apply 4 notch line filters and apply previously calculated filters
    return: variance in the given interval by seglength
    """
    dat_noth_filtered = mne.filter.notch_filter(x=dat_, Fs=sample_rate, trans_bandwidth=7,
            freqs=np.arange(line_noise, 4*line_noise, line_noise),
            fir_design='firwin', verbose=False, notch_widths=1,filter_length=dat_.shape[0]-1)

    filtered = np.zeros(filter_fun.shape[0])
    for filt in range(filter_fun.shape[0]):
        filtered[filt] = np.var(scipy.signal.convolve(filter_fun[filt,:], 
                                               dat_noth_filtered, mode='same')[-seglengths[filt]:])
    return filtered