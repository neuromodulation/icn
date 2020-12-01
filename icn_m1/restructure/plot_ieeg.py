import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

def get_epochs(dat_filtered, y_tr, epoch_len, sfreq, threshold=0):
    """Return epoched data.

    Keyword arguments
    -----------------
    dat_filtered (Numpy array) : array of extracted features in the shape (samples,channels,features)
    y_tr (Numpy array) : array of labels e.g. ones for movement and zeros for no movement or baseline corr. rotameter data
    sfreq : sampling frequency of data
    epoch_len (int) : length of epoch
    threshold (int/float) : threshold to be used for identifying events (default=0 for y_tr with only ones and zeros)

    Returns
    -------
    filtered_epoch (Numpy array) : array of epoched ieeg data with shape (epochs,samples,channels,features)
    y_arr (Numpy array) : array of epoched event label data with shape (epochs,samples)
    """

    epoch_lim = int(epoch_len * sfreq)
    ind_mov = np.where(np.diff(np.array(y_tr>threshold)*1) == 1)[0]
    low_limit = ind_mov > epoch_lim
    up_limit = ind_mov < y_tr.shape[0]-epoch_lim
    ind_mov = ind_mov[low_limit & up_limit]
    filtered_epoch = np.zeros([ind_mov.shape[0], epoch_lim*2, dat_filtered.shape[1], dat_filtered.shape[2]])
    y_arr = np.zeros([ind_mov.shape[0],int(epoch_lim*2)])
    for idx, i in enumerate(ind_mov):
        filtered_epoch[idx,:,:,:] = dat_filtered[i-epoch_lim:i+epoch_lim,:,:]
        y_arr[idx,:] = y_tr[i-epoch_lim:i+epoch_lim]
    return filtered_epoch, y_arr

def plot_feat(data, label_ar, subject, chans, feats, sfreq, epoch_len, xlim_l, xlim_h, print_plot=False, outpath=None):
    """Plot features for each channel, time-locked at onset of events and averaged over trials.

    Keyword arguments
    -----------------
    data (Numpy array) : data in the shape (samples,channels,features)
    label_ar (Numpy array) : array containing labels for all samples (e.g. 1 is movement, 0 is no movement)
    chans (list) : list of channel names
    feats (list) : list of feature names
    sfreq (int) : sampling frequency of data
    epoch_len (int) : length of epochs to be created
    xlim_l (int/float) : lower limit in seconds of epochs to be plotted
    xlim_h (int/float) : higher limit in seconds of epochs to be plotted
    print_plot (boolean) : save figure as .png (default=False)
    outpath ('string') : folder path to save figure

    Returns
    -------
    None
    """

    feat_concat, mov_concat = get_epochs(data, label_ar, threshold=0, epoch_len=epoch_len, sfreq=sfreq)

    mean_feat = feat_concat.mean(axis=0)
    for ch in range(mean_feat.shape[1]):
        for feat in range(mean_feat.shape[2]):
            mean_feat[:,ch,feat] = stats.zscore(mean_feat[:,ch,feat])

    plt.style.use('dark_background')
    fig = plt.figure(dpi=300, figsize=(5,15))
    xlab = np.arange(-epoch_len, epoch_len+1, 1, dtype=int)
    xlim1, xlim2= (epoch_len+xlim_l)*sfreq, (epoch_len+xlim_h)*sfreq

    for i, chan in enumerate(chans):
        ax = fig.add_subplot(len(chans),1,i+1)
        plt.title(chan, fontsize='small')
        plt.imshow(mean_feat[:,i,:].T, aspect='auto', interpolation=None)
        cbar = plt.colorbar()
        cbar.set_label('Power [Z-score]', fontsize='small')
        plt.xticks(np.arange(0, (epoch_len*2+1)*sfreq, sfreq), xlab, fontsize='small')
        plt.yticks(range(mean_feat.shape[2]),feats, fontsize='small')
        plt.xlim(xlim1, xlim2)
        plt.gca().invert_yaxis()

    fig.suptitle('sub-' + subject + ': Features \n', fontsize='medium')
    plt.xlabel('Time [s]', fontsize='small')
    fig.tight_layout()
    if print_plot == True:
        fig.savefig(outpath + 'Features' + '.png')
    plt.show()
