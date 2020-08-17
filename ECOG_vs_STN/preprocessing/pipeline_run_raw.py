#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:30:16 2020

@author: victoria
"""
## check new offline pipeline_

import sys
sys.path.insert(1, '/home/victoria/icn/icn_m1/')
import filter
import IO
import projection
import online_analysis
import offline_analysis
import preprocessing
import numpy as np
import json
import os
import pickle 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #for 3D plotting
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
import itertools
import mne
mne.set_log_level(verbose='warning') #to avoid info at terminal
from collections import Counter
import multiprocessing
import settings
#%%
VICTORIA = True

settings = {}

if VICTORIA is True:
   
    settings['BIDS_path'] = "/mnt/Datos/BML_CNCRS/Data_BIDS_new/"
    settings['out_path'] = "/mnt/Datos/BML_CNCRS/Data_processed/Derivatives/Raw_pipeline/"
else:
    settings['BIDS_path'] = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\"
    settings['out_path'] = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\gen_p_files\\"

settings['resamplingrate']=10
settings['max_dist_cortex']=20
settings['max_dist_subcortex']=20
settings['normalization_time']=30
settings['frequencyranges']=[[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]]
settings['seglengths']=[1, 2, 2, 3, 3, 3, 10, 10, 10]
settings['num_patients']=['000', '001', '004', '005', '006', '007', '008', '009', '010', '013', '014']


settings['BIDS_path']=settings['BIDS_path'].replace("\\", "/")
settings['out_path']=settings['out_path'].replace("\\", "/")


for s in range(1,len(settings['num_patients'])):   
    subject_path=settings['BIDS_path'] + 'sub-' +settings['num_patients'][s]
           
    subfolder=IO.get_subfolders(subject_path)
           
    vhdr_files=IO.get_files(subject_path, subfolder)
    
    len(vhdr_files)
    for f in range(len(vhdr_files)):
        #%% 5. get info and files for the specific subject/session/run
        
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
        
        #check session
        sess_right = IO.sess_right(sess)
        print(sess_right)
        
        # read channels info
        used_channels = IO.read_M1_channel_specs(vhdr_file[:-10])

        #used_channels = IO.read_used_channels() #old
        print(used_channels)
        
        # extract used channels/labels from brainvision file, split up in cortex/subcortex/labels
        #dat_ is a dict
        dat_ = IO.get_dat_cortex_subcortex(bv_raw, ch_names, used_channels)
        ind_cortex=dat_['ind_cortex']
        ind_subcortex=dat_['ind_subcortex']
        dat_ECOG=dat_['dat_cortex']
        dat_MOV=dat_['dat_label']
        dat_STN=dat_['dat_subcortex']
        
               
        
        #%% 6. rereference
        new_data, ch_names =preprocessing.rereference(run_string=vhdr_file[:-10], bv_raw=bv_raw)
       
        #%% filtering
        seglengths = settings['seglengths']
        
        # read line noise from participants.tsv
        line_noise = IO.read_line_noise(settings['BIDS_path'],subject)
        # if sf>1000:
        #     filter_len=sf
        # else:
        #     filter_len=1001
        # get the lenght of the recording signals
        recording_length = bv_raw.shape[1] 
        
        #resample
        normalization_samples = settings['normalization_time']*settings['resamplingrate']
        new_num_data_points = int((recording_length/sf)*settings['resamplingrate'])
    
        # downsample_idx states the original brainvision sample indexes are used
        downsample_idx = (np.arange(0,new_num_data_points,1)*sf/settings['resamplingrate']).astype(int)
        
        # get filter coef?
        
        filter_fun = filter.calc_band_filters(settings['frequencyranges'], sample_rate=sf)
    
        offset_start = int((sf/seglengths[0]) / (sf/settings['resamplingrate']))
      
        rf_data_norm= offline_analysis.run(sf, settings['resamplingrate'], np.asarray(seglengths), settings['frequencyranges'], downsample_idx, bv_raw, line_noise, \
                      sess_right, dat_, new_num_data_points, vhdr_file[:-10], normalization_samples, filter_fun, project=False)
            
        # data_=offline_analysis.create_continous_epochs(sf, settings['resamplingrate'], offset_start, settings['frequencyranges'], downsample_idx, line_noise, \
        #               dat_ECOG, filter_fun, new_num_data_points, Verbose=False)
        #%%ipsi o contralateral mov
            
        label_channels = np.array(ch_names)[used_channels['labels']]
        
        wl=int(recording_length/(new_num_data_points))
        mov_ch=int(len(dat_MOV)/2)
        con_true = np.empty(mov_ch, dtype=object)
        onoff=np.zeros(np.size(dat_MOV[0][sf:-1:wl]))


        #only contralateral mov
        for m in range(mov_ch):
            #right session
            if sess_right is True:
                if 'RIGHT' in label_channels[m]:
                    con_true[m]=False
                else:
                    con_true[m]=True

            #left session        
            else:
                if 'RIGHT' in label_channels[m]:
                    con_true[m]=True

                else:
                    con_true[m]=False
                    
           

            for m in range(mov_ch):
            
                target_channel_corrected=dat_MOV[m+mov_ch][sf:-1:wl] 

                onoff[target_channel_corrected>0]=1
            
                if m==0:
                    mov=target_channel_corrected
                    onoff_mov=onoff
                else:
                    mov=np.vstack((mov,target_channel_corrected))
                    onoff_mov=np.vstack((onoff_mov,onoff))
            
          
  #%%save data
        run_ = {
            "vhdr_file" : vhdr_file,
            "resamplingrate" : settings['resamplingrate'],
            "subject" : subject, 
            "run" : run, 
            "sess" : sess, 
            "sess_right" :  sess_right, 
            "used_channels" : used_channels, 
            "fs" : sf, 
            "line_noise" : line_noise, 
            "seglengths" : seglengths, 
            "normalization_samples" : normalization_samples, 
            "new_num_data_points" : new_num_data_points, 
            "downsample_idx" : downsample_idx, 
            "filter_fun" : filter_fun, 
            "offset_start" : offset_start, 
            "rf_data_median" : rf_data_norm, 
            "label_baseline_corrected" : mov, 
            "label" : onoff_mov, 
            "label_con_true" : con_true,        
        }

        out_path = os.path.join(settings['out_path'],'sub_' + subject + '_sess_' + sess + '_run_' + run + '.p')
    
        with open(out_path, 'wb') as handle:
            pickle.dump(run_, handle, protocol=pickle.HIGHEST_PROTOCOL)    

                                

  