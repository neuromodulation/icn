import importlib
import json
import os
import pickle
import sys
import time

from matplotlib import pyplot as plt
import mne
import mne_bids
import numpy as np
import pandas as pd

import features
import IO
import label_normalization
import plot_ieeg
import run_analysis

#### INPUTS TO PIPELINE 
# path of BIDS_run file to analyse 
# save as mat / p 

#### READ settings
with open(os.path.join('settings/settings-RK.json'), 'rb') as f:
    settings = json.load(f)

#### Find all iEEG files in BIDS Path
ieeg_files = IO.get_all_files(settings['bids_path'], suffix=[".vhdr", ".edf"], get_bids=True,
                              bids_root=settings['bids_path'], prefix=["SelfpacedRotation"], verbose=True)

#### SPECIFY iEEG file to read (INPUT to pipeline.py)
run_file_to_read = ieeg_files[1]

### WRITE M1.tsv if necessary
IO.write_m1(run_file_to_read, cortex_ref='average', subcortex_ref='', return_dataframe=False, overwrite=False)

### READ M1.tsv
PATH_M1 = run_file_to_read.copy().update(root=os.path.join(run_file_to_read.root,'derivatives'), suffix='channels')
df_M1 = pd.read_csv(PATH_M1, sep="\t")

### READ BIDS data, sampling frequency,line noise
raw = mne_bids.read_raw_bids(run_file_to_read)
ieeg_raw = raw.get_data()
ch_names = raw.info['ch_names']
sfreq = raw.info['sfreq']
fs = int(np.ceil(sfreq))
line_noise = raw.info['line_freq']

### READ Coordinates
df_coord = pd.read_csv(os.path.join(os.path.dirname(run_file_to_read), "sub-"+subject+"_electrodes.tsv"), sep="\t")

### CALCULATE filter
filter_fun = features.calc_band_filters(settings['frequencyranges'], sample_rate=fs, filter_len='1000ms')

### APPEND label to end of array if desired
events, event_id = mne.events_from_annotations(raw, event_id={'Movement_Onset': 1, 'Movement_End':-1})
labels = label_normalization.generate_continous_label_array(ieeg_raw[0], events)
ieeg_raw = np.vstack((ieeg_raw, labels))

### DEFINE generator
def ieeg_raw_generator(ieeg_raw, df_M1, settings, fs, include_label=False):
    """[summary]

    Args:
        ieeg_raw (np array): shape (channels, time)
        df_M1 (pandas dataframe): information for rereferencing etc. from M1.tsv file
        settings (dict): general settings from settings.json
        fs (float): sampling frequency
        include_label (boolean): True, if cleaned label has been appended to end of ieeg_raw

    Yields:
        np.array: new batch for run function of full segment length shape
    """

    cnt_fsnew = 0
    # offset_start: size of highest segment length, needs to be skipped at the start to have same feature size
    offset_start = fs // 1
    fs_new = settings["resamplingrate"]
    used_idx = df_M1[(df_M1["used"] == 1) & (df_M1["target"] == 0)].index
    if include_label:
        used_idx = used_idx.append(pd.Index([-1]))

    for cnt in range(ieeg_raw.shape[1]):
        if cnt < offset_start:
            cnt_fsnew += 1
            continue

        cnt_fsnew += 1
        if cnt_fsnew >= (fs / fs_new):
            cnt_fsnew = 0
            yield ieeg_raw[used_idx, cnt - offset_start:cnt]

### INITIALIZE generator
gen_ = ieeg_raw_generator(ieeg_raw[:,:], df_M1, settings, fs) # clip for timing reasons

### CALL run function
start = time.time()
data_features, label = run_analysis.run(gen_, settings, df_M1, fs, line_noise, filter_fun, use_mean=True,normalize=True,
                                        methods=['bandpass', 'mobility', 'complexity'], include_label=True)
end = time.time()
print(f'{round(end - start, 3)} seconds elapsed.')

### PLOT results of selected channel
channel = "ECOG_AT_SM_L_4"
channel_ind = int(df_M1[df_M1["name"] == channel].index.values)
plt.imshow(data_features[:, channel_ind, :].T, aspect='auto')
plt.colorbar()
plt.show()

### PLOT epochs if label is given
used_idx = df_M1[(df_M1["used"] == 1) & (df_M1["target"] == 0)].index.values
ch_names = df_M1["name"][used_idx]
plot_ieeg.plot_feat(data=data_features[:,used_idx,0:6], label_ar=label, fname=str(run_file_to_read.fpath),
                    chans=ch_names, feats=settings["featurelabels"], sfreq=settings["resamplingrate"],
                    epoch_len=8, xlim_l=-3, xlim_h=3, print_plot=True,
            outpath=run_file_to_read.copy().update(root=os.path.join(run_file_to_read.root, "derivatives")).directory)

# SAVE object
dict_out = {
    "data_features" : data_features, 
    "coord" : df_coord, 
    "fs" : fs, 
    "settings" : settings, 
    "df_M1" : df_M1, 
    "filters used" : filter_fun
}

out_path = os.path.join(settings['out_path'],'sub_' + subject + '_sess_' + sess + '_task_' + task + '_run_' + run + '.p')
with open(out_path, 'wb') as handle:
    pickle.dump(dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
