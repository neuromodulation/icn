import json
import os
import warnings
import numpy as np
import pandas as pd
import pybv
from scipy.signal import find_peaks
import IO
import label_normalization
import mne_bids

# reads saved feature file by pipeline_features 
# reads M1 file and settings 

ADD_CLEAN_LABEL_TOBIDS = False
Decimate = 10

### READ M1.tsv now from derivatives folder
PATH_M1 = r"C:\Users\ICN_admin\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\Datasets\BIDS_Berlin\derivatives\sub-002\ses-20200131\ieeg\sub-002_ses-20200131_task-SelfpacedRotationR+MedOn+StimOff_run-4_channels_M1.tsv"
df_M1 = pd.read_csv(PATH_M1, sep="\t")

### READ settings
with open('settings\\settings.json', 'rb') as f:
        settings = json.load(f)

# SPECIFY iEEG file to read (INPUT to pipeline.py)
ieeg_files = IO.get_all_files(settings['BIDS_path'], [".vhdr", ".edf"])  # all files
run_file_to_read = r'C:\Users\ICN_admin\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\Datasets\BIDS_Berlin\sub-002\ses-20200131\ieeg\sub-002_ses-20200131_task-SelfpacedRotationR_acq-MedOn+StimOff_run-4_ieeg.vhdr'

entities = mne_bids.get_entities_from_fname(run_file_to_read)
bids_path = mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], task=entities["task"], \
    run=entities["run"], acquisition=entities["acquisition"], datatype="ieeg", root=settings["BIDS_path"])

raw_arr = mne_bids.read_raw_bids(bids_path)
ieeg_raw = raw_arr.get_data()

fs= int(np.ceil(raw_arr.info["sfreq"]))
line_noise = int(raw_arr.info["line_freq"])

# ESTIMATE baseline correction
label_clean = np.empty((sum(df_M1['target'] == 1), ieeg_raw.shape[1]),
                       dtype=object)
label_onoff = np.empty((sum(df_M1['target'] == 1), ieeg_raw.shape[1]),
                       dtype=object)
ch_names_new = raw_arr.ch_names

for i, m in enumerate(df_M1[df_M1['target'] == 1].index.tolist()):
    # check if data should be flipped
    sign = 1
    if abs(min(ieeg_raw[m])) > max(ieeg_raw[m]):
        sign = -1
    target_channel_corrected, onoff, raw_target_channel = label_normalization.baseline_correction(
        y=sign * ieeg_raw[m],
        method='baseline_als', param=[1e3, 1e-4], thr=1.2e-1,
        normalize=True, decimate=Decimate, verbose=False)
    # check detected picks and true picks
    true_picks, _ = find_peaks(raw_target_channel, height=0, distance=0.5 * fs)
    predicted_picks, _ = find_peaks(onoff)
    if len(true_picks) != len(predicted_picks):
        warnings.warn('Check the baseline parameters and threshold, it seems they should be optimized.')
    if Decimate != 1:
        events = label_normalization.create_events_array(onoff, ieeg_raw[m])
        label = label_normalization.generate_continous_label_array(ieeg_raw[m], events)
    else:
        events = label_normalization.create_events_array(onoff, ieeg_raw[m], 1)
        label = onoff
    label_clean[i] = target_channel_corrected
    label_onoff[i] = label
    # naming
    label_name = df_M1[(df_M1["target"] == 1)]["name"][m]
    # change channel info
    ch_names_new.append(label_name + '_CLEAN')

# STACK onto data when clean
data = np.vstack((ieeg_raw, label_clean))

if ADD_CLEAN_LABEL_TOBIDS:  # --> all this should be checked
    file_name = run_file_to_read[:-5]
    out_path = settings['BIDS_path'] + 'sub-' + entities["subject"] + '/ses-' + entities["session"] + '/ieeg'
    pybv.write_brainvision(data=data, sfreq=fs, ch_names=ch_names_new,
                           fname_base=file_name,
                           folder_out=out_path,
                           events=None, resolution=1e-7,
                           fmt='binary_float32', meas_date=None)
    # modify channels info --> this part should be discussed 
# READ feature file and write corrected baseline as additional key

# call plot_ieeg --> save figure, maybe create new run folder
