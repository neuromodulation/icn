#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:44:10 2020

@author: victoria
"""
#%%
from mne_bids import write_raw_bids, make_bids_basename
import mne
import numpy as np
#import pybv
import sys
sys.path.insert(1, '/home/victoria/icn/icn_m1/')
import IO
import offline_analysis
import pybv
import os
import pandas as pd
#%%
VICTORIA = True

settings = {}

if VICTORIA is True:
    # insert at 1, 0 is the script path (or '' in REPL)
    
    settings['BIDS_path'] = "/mnt/Datos/BML_CNCRS/Data_BIDS/"
    settings['out_path'] = "/mnt/Datos/BML_CNCRS/Data_processed/Derivatives/"
else:
    settings['BIDS_path'] = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\"
    settings['out_path'] = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\gen_p_files\\"

settings['BIDS_path']=settings['BIDS_path'].replace("\\", "/")
settings['out_path']=settings['out_path'].replace("\\", "/")

#%%
def write_all_M1_channel_files(settings):
    """

    Read all channels.tsv in the settings defined BIDS path, and write all all channels_M1.tsv files 
    --> copy all channel names from channel.tsv as name
    --> set targets to 'MOV' channels 
    --> rereference all to average 
    --> used all to 1 

    """

    BIDS_channel_tsv_files = []
    for root, dirs, files in os.walk(settings['BIDS_path']):
        for file in files:
            if file.endswith("_channels.tsv"):
                ch_file = os.path.join(root, file)
                
                df_channel = pd.read_csv(ch_file, sep="\t")
                
                df = pd.DataFrame(np.nan, index=np.arange(len(list(df_channel['name']))+2), columns=['name', 'rereference', 'used', 'target'])

                df['used'] = 1
                names=list(df_channel['name'].copy(deep=True))
                names.append(names[-2:-1][0]+'_CLEAN')
                names.append(names[-2:-1][0]+'_CLEAN')
                df['name'] =names

                ch_mov = [ch_idx for ch_idx, ch in enumerate(df['name']) if ch.startswith('MOV')]
                target = np.zeros(len(list(df['name'])))
                target[ch_mov] = 1
                df['target'] = target.astype(int)
                df['rereference'] = ['average']*len(list(df['name']))

                df.to_csv(ch_file[:-12]+'channels_M1.tsv', sep='\t')

                BIDS_channel_tsv_files.append(ch_file)
#%%
def run_vhdr_file(s):
   
    if s<10:
        subject_idx= 'sub-00' + str(s)
    else:
        subject_idx= 'sub-0' + str(s)
    
    subject_path=settings['BIDS_path'] + subject_idx
    subfolder=IO.get_subfolders(subject_path)
           
    vhdr_files=IO.get_files(subject_path, subfolder)
    vhdr_files.sort()

    for f in range(len(vhdr_files)):
        #%% get info and files for the specific subject/session/run

        vhdr_file=vhdr_files[f]
         
        #get info from vhdr_file
        subject, run, sess = IO.get_sess_run_subject(vhdr_file)
        
        print('RUNNIN SUBJECT_'+ subject+ '_SESS_'+ sess + '_RUN_' + run)

        
        #read sf
        sf=IO.read_run_sampling_frequency(vhdr_file)
        if len(sf.unique())==1: #all sf are equal
            sf=int(sf[0])
        else: 
            Warning('Different sampling freq.')      
        #read data
        bv_raw, ch_names = IO.read_BIDS_file(vhdr_file)
        used_channels = IO.read_M1_channel_specs(vhdr_file[:-10])

    
        # extract used channels/labels from brainvision file, split up in cortex/subcortex/labels
        #dat_ is a dict
        dat_ = IO.get_dat_cortex_subcortex(bv_raw, ch_names, used_channels)
        dat_MOV=dat_['dat_label']
    
        label=np.empty(np.shape(dat_MOV), dtype=object)
    
        for m in range(len(dat_MOV)):
            label_name=ch_names[used_channels['labels'][m]]           
            if subject == '016':
                target_channel_corrected, onoff, raw_target_channel=offline_analysis.baseline_correction(y=-dat_MOV[m], param=1, thr=2e-1)
            else:
                target_channel_corrected, onoff, raw_target_channel=offline_analysis.baseline_correction(y=dat_MOV[m])
    
            label[m]=target_channel_corrected
            
         
           
             #change channel info
                   
            ch_names.append(label_name+'_CLEAN')
        
        data=np.vstack((bv_raw, label))
        file_name=subject_idx+'_ses-'+sess+'_task-force_run-' +run +'_ieeg'
        out_path=subject_path+'/ses-'+sess+'/ieeg'
        pybv.write_brainvision(data, sfreq=1000, ch_names=ch_names, fname_base=file_name,
                                   folder_out=out_path,
                                   events=None, resolution=1e-7, scale_data=True,
                                   fmt='binary_float32', meas_date=None)
       
       
#add label corrected to BIDS file
write_ALL = True
if write_ALL is True:
    write_all_M1_channel_files(settings)
if __name__ == "__main__":
    
    for sub in range(17):
        run_vhdr_file(sub)
        
        
# def optimize_baseline(data):
#     """Apply Bayesian Optimization to baseline correction parameters."""
#     def function(param, thr):
              
#         return baseline_correction(param, thr, y=data)

#     optimizer = BayesianOptimization(
#         f=function,
#         pbounds={"param": (1e3, 1e5), "thr": (0.5e-1, 2e-1)},
#         random_state=1234,
#         verbose=2
#     )
#     optimizer.maximize(n_iter=10)

#     print("Final result:", optimizer.max)        
    
# optimize_baseline(data)    