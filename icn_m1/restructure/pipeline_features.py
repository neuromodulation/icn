import features
import IO
import sys
import numpy as np
import offline_analysis
import time
import json
import pandas as pd 
import os
import run_analysis
import pickle
import mne_bids

'''
Inputs to pipeline
 - run file 
 - M1_derivative file 
 - correct settings in settings/settings.json

 the M1 file describes how data should be preprocessed 
  - if a channel is used 
  - which channel is a predictor 
  - how data is rereferenced 
  - specify which channel is ECOG 
'''

### READ settings
with open('settings\\settings.json', 'rb') as f:
        settings = json.load(f)


### READ M1.tsv now from derivatives folder
PATH_M1 = r"C:\Users\ICN_admin\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\Datasets\BIDS_Berlin\derivatives\sub-002\ses-20200131\ieeg\sub-002_ses-20200131_task-SelfpacedRotationR+MedOn+StimOff_run-4_channels_M1.tsv"
df_M1 = pd.read_csv(PATH_M1, sep="\t")


### SPECIFY iEEG file to read (INPUT to pipeline.py)
ieeg_files = IO.get_all_files(settings['BIDS_path'], suffix='vhdr') # read all files
run_file_to_read = r'C:\Users\ICN_admin\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\Datasets\BIDS_Berlin\sub-002\ses-20200131\ieeg\sub-002_ses-20200131_task-SelfpacedRotationR_acq-MedOn+StimOff_run-4_ieeg.vhdr'

entities = mne_bids.get_entities_from_fname(run_file_to_read)
bids_path = mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], task=entities["task"], \
    run=entities["run"], acquisition=entities["acquisition"], datatype="ieeg", root=settings["BIDS_path"])

raw_arr = mne_bids.read_raw_bids(bids_path)
ieeg_raw = raw_arr.get_data()

fs= int(np.ceil(raw_arr.info["sfreq"]))
line_noise = int(raw_arr.info["line_freq"])

### CALCULATE filter
# well here I needed to add to the filter_len 1, need to recheck if that's every time neccessary!
filter_fun = features.calc_band_filters(settings['frequencyranges'], sample_rate=fs, filter_len=fs + 1)

### DEFINE generator
def ieeg_raw_generator(ieeg_raw, df_M1, settings, fs):
    """

    This generator function mimics online data acquisition. 
    The df_M1 selected raw channels are iteratively sampled with fs.  

    Args:
        ieeg_raw (np array): shape (channels, time)
        fs (float): 
        fs_new (float): new resampled frequency 
        offset_start (int): size of highest segmenth length, needs to be skipped at the start to have same feature size

    Yields:
        np.array: new batch for run function of full segment length shape
    """

    cnt_fsnew = 0
    offset_start = int(settings["seglengths"][0] * fs)
    fs_new = settings["resamplingrate"]
    used_idx = df_M1[(df_M1["used"] == 1) & (df_M1["target"] == 0)].index
    
    for cnt in range(ieeg_raw.shape[1]):
        if cnt < offset_start:
            cnt_fsnew +=1
            continue
        
        cnt_fsnew +=1
        if cnt_fsnew >= (fs/fs_new):
            cnt_fsnew = 0
            yield ieeg_raw[used_idx,cnt-offset_start:cnt]

### INITIALIZE generator 
gen_ = ieeg_raw_generator(ieeg_raw[:,:5000], df_M1, settings, fs) # clip for timing reasons

### CALL run function 
data_features = run_analysis.run(gen_, settings, df_M1, fs, line_noise, filter_fun, usemean_=True, normalize=True)

# SAVE object
dict_out = {
    "data_features" : data_features, 
    "info" : raw_arr.info, 
    "fs" : raw_arr.info["line_freq"],
    "sfreq" : raw_arr.info["sfreq"],
    "settings" : settings, 
    "df_M1" : df_M1, 
    "filters used" : filter_fun
}

out_path = os.path.join(settings['out_path'], os.path.basename(run_file_to_read[:-10])+'.p') # :-10 cuts off _ieeg.vhdr from file name
with open(out_path, 'wb') as handle:
    pickle.dump(dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)    
