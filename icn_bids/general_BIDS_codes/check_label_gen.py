# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:11:40 2020

@author: Pilin
"""


#%% check label generator
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/victoria/icn/icn_m1')
import IO
import offline_analysis
import numpy as np
from matplotlib import pyplot as plt
import mne
mne.set_log_level(verbose='warning') #to avoid info at terminal
from matplotlib.backends.backend_pdf import PdfPages
#%%
plt.close('all')

VICTORIA = True

settings = {}

if VICTORIA is True:
    # insert at 1, 0 is the script path (or '' in REPL)
    
    settings['BIDS_path'] = "/mnt/Datos/BML_CNCRS/Data_BIDS/"
    settings['out_path'] = "/mnt/Datos/BML_CNCRS/Data_processed/Derivatives/"
else:
    settings['BIDS_path'] = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\"
    settings['out_path'] = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\gen_p_files\\"

settings['resamplingrate']=10
settings['max_dist_cortex']=20
settings['max_dist_subcortex']=5
settings['normalization_time']=10
settings['frequencyranges']=[[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]]
settings['seglengths']=[1, 2, 2, 3, 3, 3, 10, 10, 10]


settings['BIDS_path']=settings['BIDS_path'].replace("\\", "/")
settings['out_path']=settings['out_path'].replace("\\", "/")

   
#%%
# get all vhdr files (from a subject or from all BIDS_path)
# vhdr_files=IO.get_all_vhdr_files(settings['BIDS_path'])
# vhdr_files.sort()
pp = PdfPages('Baseline_correction_output.pdf', keep_empty=False)
vhdr_files=IO.get_all_vhdr_files(settings['BIDS_path'])
vhdr_files.sort()
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
    dat_MOV=dat_['dat_label']
    onoff=np.zeros(np.size(dat_MOV[0]))

   
    # y=np.empty(dat_MOV.shape, dtype=object)
    mov_ch=int(len(dat_MOV)/2)
    for m in range(mov_ch):
            
            T=len(dat_MOV[m])/sf#total time in sec
            if subject == '016':
                dat_MOV[m]=-dat_MOV[m]
            #     t= np.arange(0.0, T-1/sf , 1/sf)
            #     # target_channel_corrected, onoff, raw_target_channel=offline_analysis.baseline_correction(y=-dat_MOV[m], param=1, thr=2e-1)
            # else:
                
            #     t= np.arange(0.0, T , 1/sf)
    
                # target_channel_corrected, onoff, raw_target_channel=offline_analysis.baseline_correction(y=dat_MOV[m], method='baseline_rope', param=1e4, thr=2e-1, normalize=True, Decimate=1, Verbose=True)
            
            t= np.arange(0.0, T , 1/sf)
            if np.size(t)>np.size(dat_MOV[1]):
                t= np.arange(0.0, T-1/sf , 1/sf) 
                
            labels_plot=['original', 'clean','onset']
            raw_target_channel=offline_analysis.NormalizeData(dat_MOV[m])
            target_channel_corrected=dat_MOV[m+mov_ch]
            onoff[target_channel_corrected>0]=1
            #zoom at the first movement
            ind_toplot=np.where(onoff==1)[0][0]
            plt.figure(f+1, figsize=(8, 5))
    
            plt.subplot(1,2,m+1)
            plt.plot(raw_target_channel[ind_toplot-10:ind_toplot+2500], label=labels_plot[0])
            plt.plot(target_channel_corrected[ind_toplot-10:ind_toplot+2500], label=labels_plot[1])
            # plt.plot(onoff[ind_toplot-10:ind_toplot+2500], label=labels_plot[2])
            plt.legend()
            plt.title('subject_'+ subject+ '_session_' +sess+ '_run'+ run)
            
            if m==mov_ch-1: plt.savefig(pp, format='pdf')
    
            
            plt.figure(f+2, figsize=(8, 5))
    
            plt.subplot(2,1,m+1)
            plt.plot(t,raw_target_channel, label=labels_plot[0])
            plt.plot(t,target_channel_corrected, label=labels_plot[1])
            # plt.plot(t,onoff, label=labels_plot[2])
            plt.legend()
            plt.title('subject_'+ subject+ '_session' +sess+ '_run'+ run)
            
            if m==mov_ch-1: plt.savefig(pp, format='pdf')
            
pp.close()

        
#%%        
# for m in range(len(dat_MOV)):
#     plt.figure(f+3, figsize=(8, 4))
#     plt.subplot()
#     plt.plot(y[m])
#     plt.legend()
#     plt.title('LABELS: subject_'+ subject+ '_session' +sess+ '_run'+ run)

#     plt.savefig(pp, format='pdf')



               
    