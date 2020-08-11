import numpy as np 
import mne 
import scipy

def calc_band_filters(f_ranges, sample_rate, filter_len=1001, l_trans_bandwidth=4, h_trans_bandwidth=4):
    """
    This function returns for the given frequency band ranges filter coefficients with with length "filter_len"
    Thus the filters can be sequentially used for band power estimation

    Parameters
    ----------
    f_ranges : TYPE
        DESCRIPTION.
    sample_rate : float
        sampling frequency.
    filter_len : int, 
        lenght of the filter. The default is 1001.
    l_trans_bandwidth : TYPE, optional
        DESCRIPTION. The default is 4.
    h_trans_bandwidth : TYPE, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    filter_fun : array
        filter coefficients stored in rows.

    """
    filter_fun = np.zeros([len(f_ranges), filter_len])

    for a, f_range in enumerate(f_ranges):
        h = mne.filter.create_filter(None, sample_rate, l_freq=f_range[0], h_freq=f_range[1], 
                            fir_design='firwin', l_trans_bandwidth=l_trans_bandwidth, 
                            h_trans_bandwidth=h_trans_bandwidth, filter_length='1000ms')

        filter_fun[a, :] = h
    return filter_fun

def apply_filter(dat_, sample_rate, filter_fun, line_noise, variance=True, seglengths=None):
    """
    For a given channel, apply 4 notch line filters and apply previously calculated filters
    

    Parameters
    ----------
    dat_ : array (ns,)
        segment of data at a given channel and downsample index.
    sample_rate : float
        sampling frequency.
    filter_fun : array
        output of calc_band_filters.
    line_noise : int|float
        (in Hz) the line noise frequency.
    seglengths : list 
        list of ints with the leght to which variance is calculated. 
        Used only if variance is set to True.
    variance : bool, 
        If True, return the variance of the filtered signal, else
        the filtered signal is returned.

    Returns
    -------
    filtered : array
        if variance is set to True: (nfb,) array with the resulted variance
        at each frequency band, where nfb is the number of filter bands used to decompose the signal
        if variance is set to False: (nfb, filter_len) array with the filtered signal
        at each freq band, where nfb is the number of filter bands used to decompose the signal

    """    
    dat_noth_filtered = mne.filter.notch_filter(x=dat_, Fs=sample_rate, trans_bandwidth=7,
            freqs=np.arange(line_noise, 4*line_noise, line_noise),
            fir_design='firwin', verbose=False, notch_widths=1,filter_length=dat_.shape[0]-1)

   
    filtered = []

    for filt in range(filter_fun.shape[0]):
        if variance:
            filtered.append(np.var(scipy.signal.convolve(filter_fun[filt,:], 
                                               dat_noth_filtered, mode='same')[-seglengths[filt]:]))
        else:
            filtered.append(scipy.signal.convolve(filter_fun[filt,:], 
                                               dat_noth_filtered, mode='same'))
                
    return np.array(filtered)