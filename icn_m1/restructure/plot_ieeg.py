import os.path as od

from matplotlib import pyplot as plt
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
from scipy import stats

from IO import get_subject_sess_task_run


def get_epochs(dat_filtered, y_tr, epoch_len, sfreq, threshold=0):
    """Return epoched data.

    Keyword arguments
    -----------------
    dat_filtered (array) : array of extracted features of shape (n_samples, n_channels, n_features)
    y_tr (array) : array of labels e.g. ones for movement and zeros for no movement or baseline corr. rotameter data
    sfreq (int/float) : sampling frequency of data
    epoch_len (int) : length of epoch in seconds
    threshold (int/float) : (Optional) threshold to be used for identifying events (default=0 for y_tr with only ones
    and zeros)

    Returns
    -------
    filtered_epoch (Numpy array) : array of epoched ieeg data with shape (epochs,samples,channels,features)
    y_arr (Numpy array) : array of epoched event label data with shape (epochs,samples)
    """

    epoch_lim = int(epoch_len * sfreq)
    ind_mov = np.where(np.diff(np.array(y_tr>threshold)*1) == 1)[0]
    print("ind_mov: ", ind_mov.shape)
    low_limit = ind_mov > epoch_lim/2
    up_limit = ind_mov < y_tr.shape[0]-epoch_lim/2
    ind_mov = ind_mov[low_limit & up_limit]
    filtered_epoch = np.zeros([ind_mov.shape[0], epoch_lim, dat_filtered.shape[1], dat_filtered.shape[2]])
    y_arr = np.zeros([ind_mov.shape[0],int(epoch_lim)])
    for idx, i in enumerate(ind_mov):
        filtered_epoch[idx,:,:,:] = dat_filtered[i-epoch_lim//2:i+epoch_lim//2,:,:]
        y_arr[idx,:] = y_tr[i-epoch_lim//2:i+epoch_lim//2]
    return filtered_epoch, y_arr


def plot_feat(data, label_ar, fname, chans, feats, sfreq, epoch_len, xlim_l, xlim_h, print_plot=False, outpath=None):
    """Plot features for each channel, time-locked at onset of events and averaged over trials.

    Keyword arguments
    -----------------
    data (array) : data of shape (n_samples, n_channels, n_features)
    label_ar (array) : array containing labels for all samples (e.g. 1 is movement, 0 is no movement)
    fname (string) : input filename (not including directory!)
    chans (list) : list of channel names
    feats (list) : list of feature names
    sfreq (int) : sampling frequency of data
    epoch_len (int) : length of epochs in seconds to be created
    xlim_l (int/float) : lower limit in seconds of epochs to be plotted
    xlim_h (int/float) : higher limit in seconds of epochs to be plotted
    print_plot (boolean) : save figure as .png (default=False)
    outpath (string/path) : folder path to save figure

    Returns
    -------
    None
    """
    
    subject, session, task, run = get_subject_sess_task_run(fname)
    feat_concat, mov_concat = get_epochs(data, label_ar, threshold=0, epoch_len=epoch_len, sfreq=sfreq)

    mean_feat = feat_concat.mean(axis=0)
    for ch in range(mean_feat.shape[1]):
        for feat in range(mean_feat.shape[2]):
            mean_feat[:,ch,feat] = stats.zscore(mean_feat[:,ch,feat])

    epoch_lim = epoch_len//2
    plt.style.use('dark_background')
    fig = plt.figure(dpi=300,figsize=(5, len(chans)*1.5)) 
    fig.suptitle('sub-' + subject + ': Features', y = 1, fontsize='medium')
    xlab = np.arange(-epoch_lim, epoch_lim+1, 1, dtype=int)
    xlim1, xlim2 = (epoch_lim+xlim_l)*sfreq, (epoch_lim+xlim_h)*sfreq

    for i, chan in enumerate(chans):
        ax = fig.add_subplot(len(chans),1,i+1)
        plt.title(chan, fontsize='small')
        plt.imshow(mean_feat[:,i,:].T, aspect='auto', interpolation=None)
        cbar = plt.colorbar()
        cbar.set_label('Power [Z-score]', fontsize='small')
        plt.xticks(np.arange(0, (epoch_len+1)*sfreq, sfreq), xlab, fontsize='small')
        plt.yticks(range(mean_feat.shape[2]),feats, fontsize='small')
        plt.xlim(xlim1, xlim2)
        plt.gca().invert_yaxis()

    plt.xlabel('Time [s]', fontsize='small')
    plt.tight_layout()
    if print_plot == True:
        fig.savefig(od.join(outpath, 'sub_' + subject +'_sess_' + session + '_task_' + task + '_run_' + run + '_features' + '.png'))
    plt.show()

def plot_raw_data(files, bids_root=None, highpass=0.1, lowpass=90, decim="auto"):
    """Plot raw ieeg data of given files structured in BIDS compatible folder.

    Args:
        bids_root (str/path): Root of BIDS folder.
        files (list of strings/BIDSPaths): List of files to be plotted. Data must be structured according to BIDS.
        highpass (int/float): Value of highpass filter for plotting. Default=0.1.
        lowpass (int/float): Value of highpass filter for plotting. Default=90.
        decim: If decimation to enhance responsiveness is desired. Default="auto".
    Returns:
        None
    """

    for file in files:
        try:
            raw = read_raw_bids(file, verbose=False)
        except RuntimeError:
            subject, session, task, run = get_subject_sess_task_run(file)
            file = BIDSPath(subject=subject, session=session, task=task, run=run, datatype="ieeg",
                                 root=bids_root)
            raw = read_raw_bids(file, verbose=False)
        raw.plot(block=True, highpass=highpass, lowpass=lowpass, decim=decim, scalings='auto', verbose=False)
