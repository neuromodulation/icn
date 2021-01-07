import features
import numpy as np

def create_events_array(onoff, raw_target_channel, sf):
    """
    Given a binary label (onoff) an arrray of events is created.

    Parameters
    ----------
    onoff : array, shape(n_samples)
        onoff array. squared signal. when up it indicates the target taks
        was being done. Output of baseline_correction
    raw_target_channel : array, shape(n_samples2)
        the raw signal which which contains the performed taks.
        Needed to estimate time of the events.
    sf : float
        sampling frequency of the raw_target_channel.
        Needed to estimate the time of the events.

    Returns
    -------
    events : array, shape(n_events, 2)
        All events that were found.
        The first column contains the event time in samples and the
        second column contains the event id.
        1= taks starts, -1=taks stops

    """

    # create time vector
    T = len(raw_target_channel)/sf
    Df = round(len(raw_target_channel)/len(onoff))

    # time onoff_signal
    t = np.arange(0.0, T-Df/sf, Df/sf)

    # diff to find up and down times
    onoff_dif = np.diff(onoff)
    # create time info
    index_start = onoff_dif == 1
    time_start = t[index_start]
    index_stop = onoff_dif == -1
    time_stop = t[index_stop]

    if len(time_stop) > len(time_start):
       if time_stop[0]<time_start[0]:
           time_stop=time_stop[1:]
    else:
        if time_start[-1] > time_stop[-1]:
            time_start = time_start[:-1]

    time_event = np.hstack((time_start, time_stop))
    time_event = np.sort(time_event)
    id_event = np.asarray([1, -1] * len(time_start))
    events = np.transpose(np.vstack((time_event, id_event)))

    return events

def generate_continous_label_array(L, sf, events):
    """
    Given an arrray of events, this function returns sample-by-sample
    label information of raw_date

    Parameters
    ----------
    L : float, 
        lenght (n_samples) of the corresponding signal to labelled
    sf : int, float
        sampling frequency of the raw_data
    events : array, shape(n_events,2)
        All events that were found by the function
        'create_events_array'. 
        The first column contains the event time in samples and the second
        column contains the event id.
   
    Returns
    -------
    labels : array (n_samples)
        array of ones and zeros.

    """

    labels = np.zeros(L)

    mask_start = events[:, 1] == 1
    start_event_time = events[mask_start, 0]
    mask_stop = events[:, 1] == -1
    stop_event_time = events[mask_stop, 0]

    for i in range(len(start_event_time)):
        range_up = np.arange(int(np.round(start_event_time[i]*sf)),
                             int(np.round(stop_event_time[i]*sf)))
        labels[range_up] = 1

    return labels

def create_continous_epochs(fs, fs_new, offset_start, f_ranges, downsample_idx,
                            line_noise, data_, filter_fun, new_num_data_points,
                            Verbose=False):
    """
    Create epochs from raw data.

    Parameters
    ----------
    fs : TYPE
        DESCRIPTION.
    fs_new : TYPE
        DESCRIPTION.
    offset_start : TYPE
        DESCRIPTION.
    f_ranges : TYPE
        DESCRIPTION.
    downsample_idx : TYPE
        DESCRIPTION.
    line_noise : TYPE
        DESCRIPTION.
    data_ : TYPE
        DESCRIPTION.
    filter_fun : TYPE
        DESCRIPTION.
    new_num_data_points : TYPE
        DESCRIPTION.
    Verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    rf_data : TYPE
        DESCRIPTION.

    """

    num_channels = data_.shape[0]
    num_f_bands = len(f_ranges)
    num_samples =  np.shape(filter_fun)[1]
    #
    rf_data = np.zeros([new_num_data_points-offset_start, num_channels, num_samples, num_f_bands])  # raw frequency array

    new_idx = 0

    for c in range(downsample_idx.shape[0]):
        if Verbose: 
            print(str(np.round(c*(1/fs_new),2))+' s')
        if downsample_idx[c]<downsample_idx[offset_start]:  # neccessary since downsample_idx starts with 0, wait till 1s for theta is over
            continue

        for ch in range(num_channels):    
            dat_ = data_[ch, downsample_idx[c-offset_start]:downsample_idx[c]]
            dat_filt = features.apply_filter(dat_, sample_rate=fs, filter_fun=filter_fun, line_noise=line_noise, variance=False)
            rf_data[new_idx,ch,:,:] = dat_filt.T

        
        new_idx += 1
        
    
    
    return rf_data

def create_epochs_data(data, events, sf, tmin=1, tmax=1):
    """
    this function segments data tmin sec before target onset and tmax sec
    after target onset

    Parameters
    ----------
    data : array, shape(n_channels, n_samples)
        either cortex of subcortex data to be epoched.
    events : array, shape(n_events,2)
        All events that were found by the function
        'create_events_array'. 
        The first column contains the event time in samples and the second column contains the event id.
    sf : int, float
        sampling frequency of the raw_data.
    tmin : float
        Start time before event (in sec). 
        If nothing is provided, defaults to 1.
    tmax : float
        Stop time after  event (in sec). 
        If nothing is provided, defaults to 1.
    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    X : array, shape(n_events, n_channels, n_samples)
        epoched data
    Y : array, shape(n_events, n_samples)
        sample-wise label information of data.
.

    """

    #get time_events index    
    mask_start=events[:,1]==1
    start_event_time=events[mask_start,0]
    #labels
    labels=generate_continous_label_array(data, sf, events)
    
        
    for i in range(len(start_event_time)):
        #check inputs

        if i==0:
            if tmin>start_event_time[i]:
                tmin=0
                Warning('pre_time too large. It should be lower than={:3.2f}'.format(start_event_time[i]))
                Warning('for the first run is set equal to t0')

        else:
            if tmin>start_event_time[i]:
                Warning('pre_time too large. It gets data from previous trials.')
        #<<still missing: tmax for last trail>>
            
        start_epoch=int(np.round((start_event_time[i]-tmin)*sf))
        stop_epoch=int(np.round((start_event_time[i]+ tmax)*sf))
        
        epoch=data[:,start_epoch:stop_epoch]
        #reshape data (n_events, n_channels, n_samples)
        nc, ns=np.shape(epoch)
        epoch=np.reshape(epoch,(1, nc,ns))
        label=labels[start_epoch:stop_epoch]
        if i==0:
            X=epoch
            Y=label
        else:
            X=np.vstack((X,epoch))
            Y=np.vstack((Y,label))
            
    return X, Y

def create_epochs_feature_matrix(feature_matrix, events, sf, tmin=1, tmax=1):
    """
    this function segments the feature matrix generated after the filter-bank
    analysis made on the funciton RUN. The segment are extracted
    tmin sec before target onset and tmax sec
    after target onset

    Parameters
    ----------
    feature_matrix : array, shape(n_features, n_ch, n_fb)
        feature matrix of n_features at n_fb frequency bands
        either before or after the grid-projection.
    events : array, shape(n_events,2)
        All events that were found by the function
        'create_events_array'. 
        The first column contains the event time in samples and the second 
        column contains the event id.
    sf : int, float
        sampling frequency of the feature_matrix
    tmin : float
        Start time before event (in sec). 
        If nothing is provided, defaults to 1.
    tmax : float
        Stop time after  event (in sec). 
        If nothing is provided, defaults to 1.
    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    X : array, shape(n_events, n_channels, n_bf*n_samples)
        epoched feature matrix. The features at specific time window are 
        concatenated per each frequency band
    Y : array, shape(n_events, n_bf*n_samples)
        sample-wise label information of data.
.

    """
    
    
    #get time_events index    
    mask_start=events[:,1]==1
    start_event_time=events[mask_start,0]
    #labels
    labels=generate_continous_label_array(np.transpose(feature_matrix), sf, events)
    
        
    for i in range(len(start_event_time)):
        #check inputs

        if i==0:
            if tmin>start_event_time[i]:
                tmin=0
                Warning('pre_time too large. It should be lower than={:3.2f}'.format(start_event_time[i]))
                Warning('for the first run is set equal to t0')

        else:
            if tmin>start_event_time[i]:
                Warning('pre_time too large. It gets data from previous trials.')
        #<<still missing: tmax for last trail>>
            
        start_epoch=int(np.round((start_event_time[i]-tmin)*sf))
        stop_epoch=int(np.round((start_event_time[i]+ tmax)*sf))
        
        epoch=feature_matrix[start_epoch:stop_epoch]
        #reshape data (n_events, n_channels, n_samples)
        nf, nc, nb=np.shape(epoch)
        epoch=np.reshape(epoch,(1, nc,nb*nf))
        label=labels[start_epoch:stop_epoch]
        if i==0:
            X=epoch
            Y=label
        else:
            X=np.vstack((X,epoch))
            Y=np.vstack((Y,label))
            
    return X, Y
