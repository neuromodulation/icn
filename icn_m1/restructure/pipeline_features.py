import filter
import IO
import sys
import numpy as np
import offline_analysis
import time
import json
import pandas as pd 
import os
import run_analysis

#### INPUTS TO PIPELINE 
# path of BIDS_run file to analyse 
# save as mat / p 

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

### CALCULATE filter
# well here I needed t add to the filter_len 1, need to recheck if that's every time neccessary!
filter_fun = filter.calc_band_filters(settings['frequencyranges'], sample_rate=fs, filter_len=fs+1)

### DEFINE generator
def ieeg_raw_generator(ieeg_raw, df_M1, settings, fs):
    """[summary]

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
gen_ = ieeg_raw_generator(ieeg_raw[:,:20000], df_M1, settings, fs) # clip for timing reasons

### CALL run function 
data_features = run_analysis.run(gen_, settings, df_M1, fs, line_noise, filter_fun, usemean_=True, normalize=True)

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
