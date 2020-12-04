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

# READ feature file and write corrected baseline as additional key

# call plot_ieeg --> save figure, maybe create new run folder 
