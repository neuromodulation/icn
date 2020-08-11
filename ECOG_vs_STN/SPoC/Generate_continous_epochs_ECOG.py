# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:49:39 2020

@author: VPeterson
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/victoria/icn/icn_m1')
import IO
import filter
import offline_analysis
import rereference
import numpy as np
import json
import os
import pickle 
import mne
from mne import Epochs
mne.set_log_level(verbose='warning') #to avoid info at terminal

import gc

#%%
def get_files(subject_path, subfolder, endswith='.vhdr', Verbose=True):
    """
    given an address to a subject folder and a list of subfolders, provides a list of all vhdr files
    recorded for that particular subject.
    
    To access to a particular vhdr_file please see 'read_BIDS_file'.
    
    To get info from vhdr_file please see 'get_sess_run_subject'

    
    Parameters
    ----------
    subject_path : string
    subfolder : list
    Verbose : boolean, optional

    Returns
    -------
    vhdr_files : list
        list of addrress to access to a particular vhdr_file.
        

    """
    vhdr_files=[]
    session_path=subject_path+'/'+subfolder+'/ieeg'
    for f_name in os.listdir(session_path):
        if f_name.endswith(endswith):
            vhdr_files.append(session_path+ '/' +f_name)
            if Verbose: print(f_name)
    return vhdr_files

#%%

settings = {}
settings['BIDS_path'] = "/mnt/Datos/BML_CNCRS/Data_BIDS_new/"
settings['out_path'] = "/mnt/Datos/BML_CNCRS/Spoc/"
settings['resamplingrate']=10
settings['normalization_time']=10
settings['frequencyranges']=[[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]]
settings['seglengths']=[1, 2, 2, 3, 3, 3, 10, 10, 10]
settings['num_patients']=['000', '001', '004', '005', '006', '007', '008', '009', '010', '013', '014']

settings['BIDS_path']=settings['BIDS_path'].replace("\\", "/")
settings['out_path']=settings['out_path'].replace("\\", "/")


#%%

for s in range(len(settings['num_patients'])):
    
    

    subject_path=settings['BIDS_path'] + 'sub-' + settings['num_patients'][s]
    subfolder=IO.get_subfolders(subject_path)

    for ss in range(len(subfolder)):
            
           
        vhdr_files=get_files(subject_path, subfolder[ss])
        vhdr_files.sort()
        
        if s==4 and ss==0:
            vhdr_files.pop(0)
        if s==4 and ss==1:
            vhdr_files.pop(2)
    

        len(vhdr_files)
        for f in range(len(vhdr_files)):
     
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
            ind_label=dat_['ind_label']
            ind_subcortex=dat_['ind_subcortex']
            dat_ECOG=dat_['dat_cortex']
            dat_MOV=dat_['dat_label']
            dat_STN=dat_['dat_subcortex']
            
            label_channels = np.array(ch_names)[used_channels['labels']]
    
            #%% REREFERENCE
            dat_ECOG, dat_STN =rereference.rereference(run_string=vhdr_file[:-10], data_cortex=dat_ECOG, data_subcortex=dat_STN)

            #%% FILTER AND EPOCHED DATA
            #define parameters
            seglengths = settings['seglengths']
            recording_length = bv_raw.shape[1] 
            line_noise = IO.read_line_noise(settings['BIDS_path'],subject)
            new_num_data_points = int((recording_length/sf)*settings['resamplingrate'])

            downsample_idx = (np.arange(0,new_num_data_points,1)*sf/settings['resamplingrate']).astype(int)
            filter_fun = filter.calc_band_filters(settings['frequencyranges'], sample_rate=sf)
            offset_start = int((sf/seglengths[0]) / (sf/settings['resamplingrate']))
            
            
            data=offline_analysis.create_continous_epochs(sf, settings['resamplingrate'], offset_start, settings['frequencyranges'], downsample_idx, line_noise, \
                      dat_ECOG, filter_fun, new_num_data_points, Verbose=False)
                      
               
            
            mov_ch=int(len(dat_MOV)/2)
            con_true = np.empty(mov_ch, dtype=object)

            onoff=np.zeros(np.size(dat_MOV[0][::100][10:]))

            for m in range(mov_ch):
            
                target_channel_corrected=dat_MOV[m+mov_ch][::100]
                target_channel_corrected=target_channel_corrected[10:]



                onoff[target_channel_corrected>0]=1
            
                if m==0:
                    mov=target_channel_corrected
                    onoff_mov=onoff
                else:
                    mov=np.vstack((mov,target_channel_corrected))
                    onoff_mov=np.vstack((onoff_mov,onoff))
    
                
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
            
            y_con=np.squeeze(mov[con_true==True])
            y_ips=np.squeeze(mov[con_true==False])
            
            onoff_mov_con=np.squeeze(onoff_mov[con_true==True])
            onoff_mov_ips=np.squeeze(onoff_mov[con_true==False])
  
       
            
            #%% save
            sub_ = {
                "ch_names" : ch_names, 
                "subject" : subject, 
                "used_channels" : used_channels, 
                "fs" : sf, 
                "line_noise" : line_noise, 
                "label_con" : y_con, 
                "label_ips" : y_ips,         
                "onoff_con" : onoff_mov_con, 
                "onoff_ips" : onoff_mov_ips, 
                "epochs" : data,
                "session": sess,
                "run": run,
              
                
            }
            
            out_path = os.path.join(settings['out_path'],'ECOG_epochs_sub_' + subject +'_sess_' +sess + '_run_'+ run + '.p')
            
            with open(out_path, 'wb') as handle:
                pickle.dump(sub_, handle, protocol=pickle.HIGHEST_PROTOCOL)   
