import os
import warnings

import cvxpy as cp
from mne_bids import read_raw_bids, BIDSPath
import numpy as np
import pandas as pd
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve


def normalize_data(data):
    """Normalize input data. Aux function of baseline_correction."""
    minv = np.min(data)
    maxv = np.max(data)
    data_new = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data_new, minv, maxv


def de_normalize_data(data, minv, maxv):
    """De-normalize input data. Aux function of baseline_correction."""
    data_new = (data + minv) * (maxv - minv)
    return data_new


def baseline_als(y, lam, p, niter=10):
    """
    Baseline drift correction based on [1].

    A linear problem is solved: (W + lam*D'*D)z=Wy, where W=diag(w) and
    D=second order diff. matrix.

    Parameters
    ----------
    y : array
        raw signal to be cleaned
    lam : float
        reg. parameter. lam > 0
    p : int
        asymmetric parameter. Value in (0 1).
    niter : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    z : array
        the basesline to be subtracted.

    References
    ----------
    [1] P. H. C. Eilers, H. F. M. Boelens, Baseline correction with asymmetric
    least squares smoothing, Leiden University Medical Centre report, 2005.
    """
    length = y.shape[0]
    d = sparse.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2))
    w = np.ones(length)
    for i in range(niter):
        W = sparse.spdiags(w, 0, length, length)
        Z = W + lam * d.dot(d.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def baseline_rope(y, lam=1):
    """
    Baseline drift correction based on [1].

    Problem to Solve min |y-b| + lam*(diff_b)^2, s.t. b<=y.

    Parameters
    ----------
    y : array
        raw signal to be cleaned..
    lam : float (Default is 1)
        reg. parameter. lam > 0

    Returns
    -------
    z : array
        the basesline to be subtracted.

    References
    ----------
    [1] Xie, Z., Schwartz, O., & Prasad, A. (2018). Decoding of finger
    trajectory from ECoG using deep learning. Journal of neural engineering,
    15(3), 036009.
    """
    b = cp.Variable(y.shape)
    objective = cp.Minimize(cp.norm(y - b, 2) + lam * cp.sum_squares(cp.diff(b, 1)))
    constraints = [b <= y]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="SCS")
    z = b.value

    return z


def baseline_correction(y, method='baseline_rope', param=1e4, thr=2e-1,
                        normalize=True, decimate=1, niter=10, verbose=True):
    """
    Baseline correction is applied to the label.

    Parameters
    ----------
    y : array/np.array
        raw signal to be corrected
    method : string, optional
        two possible methods for baseline correction are allowed 'baseline_rope'
        and 'baseline_als'. See documentation of each methods. The default is
        'baseline_rope'.
    param : number or array of numbers, optional
        parameters needed in each optimization methods. If baseline_rope is
        being used, "param" refers to the regularization parameter.
        If baseline_als is being used  "param" should be a 2-lenght array where
        the first value is the regularization parameter and the second is the
        weigthed value. The default is [1e2, 1e-4].
    thr : number, optional
        threshold value in each small variation between trails could still
        remains after baseline elimination. The default is 1e-1.
    normalize : boolean, optional
        if normalize is True the original signal as well as the output
        corrected signal will be scalled between 0 and 1. The default is True.
    decimate: number, optinal
        before baseline correction it might be necessary to downsample the
        original raw signal. We recommend to do this step when long processing
        times are willing to be avoided. The default is 1, i.e. no decimation.
    verbose: boolean, optional
        The default is True.

    Returns
    -------
    y_corrected: signal with baseline correction
    onoff: squared signal useful for onset target evaluation.
    y: original signal
    """
    if decimate != 1:
        if verbose:
            print('>>Signal decimation is being done')
        y = signal.decimate(y, decimate)

    if method == 'baseline_als' and np.size(param) != 2:
        raise ValueError("If baseline_als methods is desired, param should be a"
                         "2 length object")
    if method == 'baseline_rope' and np.size(param) > 1:
        raise ValueError("If baseline_rope methods is desired, param should be"
                         " a number")

    if method == 'baseline_als':
        if verbose:
            print('>>baseline_als is being used')
        z = baseline_als(y, lam=param[0], p=param[1], niter=niter)
    else:
        if verbose:
            print('>>baseline_rope is being used')
        z = baseline_rope(y, lam=param)

    # subtract baseline
    y_corrected = y - z

    # aux step: normalize to eliminate interferation
    y_corrected, minv, maxv = normalize_data(y_corrected)

    # eliminate interferation
    y_corrected[y_corrected < thr] = 0
    # create on-off signal
    onoff = np.zeros(np.size(y_corrected))
    onoff[y_corrected > 0] = 1

    if normalize:
        y, Nan, Nan = normalize_data(y)
    else:
        y_corrected = de_normalize_data(y_corrected, minv, maxv)
    return y_corrected, onoff, y


def clean_labels(bids_file, decimate=10, method='baseline_rope', param=1e4, thr=2e-1, niter=10):
    """Baseline correct label array and stack cleaned label array onto raw data.

    """
    raw = read_raw_bids(bids_file, verbose=False)
    fs = int(np.ceil(raw.info['sfreq']))
    ieeg_raw = raw.get_data()
    path_M1 = bids_file.copy().update(root=os.path.join(bids_file.root, "derivatives"), suffix="channels")
    df_M1 = pd.read_csv(path_M1, sep="\t")
    label_clean_list = []
    label_onoff_list = []
    events_list = []
    ch_names_new = raw.ch_names.copy()

    for i, m in enumerate(df_M1[df_M1['target'] == 1].index.tolist()):
        # check if data should be flipped
        sign = 1
        if abs(min(ieeg_raw[m])) > max(ieeg_raw[m]):
            sign = -1
        target_channel_corrected, onoff, raw_target_channel = baseline_correction(y=sign * ieeg_raw[m], method=method,
                                                                                  param=param, thr=thr, normalize=True,
                                                                                  decimate=decimate, niter=niter,
                                                                                  verbose=False)
        # check detected picks and true picks
        true_peaks, _ = signal.find_peaks(raw_target_channel, height=0, distance=0.5 * fs)
        predicted_peaks, _ = signal.find_peaks(onoff)
        print('True peaks: ' + str(len(true_peaks)) + ', predicted peaks: ' + str(len(predicted_peaks)))
        if len(true_peaks) != len(predicted_peaks):
            warnings.warn('Check the baseline parameters and threshold, it seems they should be optimized.')

        if decimate != 1:
            events = create_events_array(onoff, ieeg_raw[m])
            label = generate_continous_label_array(ieeg_raw[m], events)
        else:
            events = create_events_array(onoff, ieeg_raw[m], 1)
            label = onoff
        label_clean_list.append(target_channel_corrected)
        label_onoff_list.append(label)
        events_list.append(events)
        # naming
        label_name = df_M1[(df_M1["target"] == 1)]["name"][m]
        # change channel info
        ch_names_new.append(label_name + '_CLEAN')
    label_clean = np.array(label_clean_list)
    label_onoff = np.array(label_onoff_list)
    return label_clean, label_onoff, ch_names_new, events_list


def generate_continous_label_array(raw_data, events, sfreq=1):
    """
    given an array of events, this function returns sample-by-sample
    label information of raw_date
    Parameters
    ----------
    raw_data : array-like,
        Corresponding signal to labelled with shape (n_samples).
    events : array, shape(n_events,2)
        Events that were found by the function 'create_events_array'.
        The first column contains the event time in samples and the second column contains the event-id.
    sfreq : float/int
        Sampling frequency. Optional, default=1

    Returns
    -------
    labels : array (n_samples)
        array of ones and zeros.
    """
    labels = np.zeros(raw_data.shape[0])

    mask_start = events[:, -1] == 1
    start_event_time = events[mask_start, 0]
    mask_stop = events[:, -1] == -1
    stop_event_time = events[mask_stop, 0]

    for i in range(len(start_event_time)):
        range_up = np.arange(int(np.round(start_event_time[i] * sfreq)), int(np.round(stop_event_time[i] * sfreq)))
        labels[range_up] = 1
    return labels


def create_events_array(onoff, raw_target_data, sf=1):
    """Create array indicating start and stop of events from squared signal of zeros and ones.

    Parameters
    ----------
    onoff : array, shape(n_samples)
        Squared signal of zeros and ones. When up it indicates the target task was being done.
        Output of baseline_correction.
    raw_target_data : array, shape(n_samples)
        The raw signal which which contains the performed task. Needed to estimate time of the events.
    sf : int/float (Optional)
        The sampling frequency of input data. If not 1, events_array will be returned as time (s) and not sample.
    Returns
    -------
    events : array, shape(n_events, 2)
        All events that were found. The first column contains the event time in samples and the second column
            contains the event-id. Event-id: 1 = task starts, -1 = task stops
    """

    # create time vector
    tf = len(raw_target_data) / sf
    df = len(raw_target_data) / len(onoff)
    af = round(tf - df / sf)

    # time onoff_signal
    t = np.arange(0.0, af, df / sf)

    # diff to find up and down times
    onoff_dif = np.diff(onoff)
    # create time info
    index_start = onoff_dif == 1
    time_start = t[index_start]
    index_stop = onoff_dif == -1
    time_stop = t[index_stop]

    if len(time_stop) > len(time_start):
        if time_stop[0] < time_start[0]:
            time_stop = time_stop[1:]
    else:
        if time_start[-1] > time_stop[-1]:
            time_start = time_start[:-1]

    time_event = np.hstack((time_start, time_stop))
    time_event = np.sort(time_event)

    id_event = np.asarray([1, -1] * len(time_start))

    events = np.transpose(np.vstack((time_event, np.zeros(time_event.shape[0]), id_event))).astype(int)
    return events
