import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pybv
from scipy.signal import find_peaks

import IO
import label_normalization

# reads saved feature file by pipeline_features 
# reads M1, settings, dat 

### READ M1.tsv 
PATH_M1 = "C:\\Users\\ICN_admin\\Charité - Universitätsmedizin Berlin\\Interventional Cognitive Neuromodulation - Data\\Datasets\\BIDS Berlin\\sub-002\\ses-20200131\\ieeg\\sub-002_ses-20200131_task-selfpacedrotation202001310001_run-4_channels_M1.tsv"
df_M1 = pd.read_csv(PATH_M1, sep="\t") 

### READ settings
with open('settings\\settings.json', 'rb') as f:
        settings = json.load(f)
settings["BIDS_path"] = "C:\\Users\\ICN_admin\\Charité - Universitätsmedizin Berlin\\Interventional Cognitive Neuromodulation - Data\\Datasets\\BIDS Berlin"

### SPECIFY iEEG file to read (INPUT to pipeline.py)
ieeg_files = IO.get_all_ieeg_files(settings['BIDS_path']) # all files...
run_file_to_read = "C:\\Users\\ICN_admin\\Charité - Universitätsmedizin Berlin\\Interventional Cognitive Neuromodulation - Data\\Datasets\\BIDS Berlin\\sub-002\\ses-20200131\\ieeg\\sub-002_ses-20200131_task-selfpacedrotation202001310001_run-4_ieeg.vhdr"
subject, sess, task, run = IO.get_subject_sess_task_run(os.path.basename(run_file_to_read))

### READ BIDS data
ieeg_raw, ch_names = IO.read_BIDS_file(run_file_to_read)

### READ Coordinates
df_coord = pd.read_csv(os.path.join(os.path.dirname(run_file_to_read), \
                                    "sub-"+subject+"_electrodes.tsv"), sep="\t")

### READ sampling frequency (could be done by mne_bids.read_raw_bids)
fs=IO.read_run_sampling_frequency(run_file_to_read)[0] 
fs = int(np.ceil(fs))

### READ line noise (could be done by mne_bids.read_raw_bids)
line_noise = IO.read_line_noise(settings['BIDS_path'], subject) # line noise is a column in the participants.tsv

# estimate baseline correction
label_clean = np.empty((sum(df_M1['target'] == 1), ieeg_raw.shape[1]),
                       dtype=object)
ch_names_new = ch_names.copy()
ADD_CLEAN_LABEL_TOBIDS = False
for i, m in enumerate(df_M1[df_M1['target'] == 1].index.tolist()):
    # check if data should be flipped
    sign = 1
    if abs(min(ieeg_raw[m])) > max(ieeg_raw[m]):
        sign = -1
    target_channel_corrected, onoff, raw_target_channel=label_normalization.baseline_correction(y=sign * ieeg_raw[m], param=1, thr=2e-1, normalize=False)
    # check detected picks and true picks
    true_picks, _ = find_peaks(raw_target_channel, height=0, distance=0.5 * fs)
    predicted_picks, _ = find_peaks(onoff)
    if len(true_picks) != len(predicted_picks):
        warnings.warn('Check the baseline parameters, it seems they should be optimized')

    label_clean[i] = target_channel_corrected
    # naming
    label_name = df_M1[(df_M1["target"] == 1)]["name"][m]
    # change channel info
    ch_names_new.append(label_name+'_CLEAN')

# when clean, stack to data
data = np.vstack((ieeg_raw, label_clean))

if ADD_CLEAN_LABEL_TOBIDS: # --> all this should be checked 
    file_name = run_file_to_read[:-5]
    out_path = settings['BIDS_path'] + 'sub-' + subject + '/ses-'+sess+'/ieeg'
    pybv.write_brainvision(data, sfreq=fs, ch_names=ch_names_new,
                           fname_base=file_name,
                           folder_out=out_path,
                           events=None, resolution=1e-7, scale_data=True,
                           fmt='binary_float32', meas_date=None)
    # modify channels info --> this part should be discussed 
# READ feature file and write corrected baseline as additional key

# call plot_ieeg --> save figure, maybe create new run folder 
