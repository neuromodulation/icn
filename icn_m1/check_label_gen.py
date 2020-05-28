# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:11:40 2020

@author: Pilin
"""


#%% check label generator
import IO
import offline_analysis
import numpy as np
from matplotlib import pyplot as plt
import mne
mne.set_log_level(verbose='warning') #to avoid info at terminal

from matplotlib.backends.backend_pdf import PdfPages

plt.close('all')

#%%
#1. Read settings
settings = IO.read_settings('mysettings')
#2. write _channels_MI file
IO.write_all_M1_channel_files()
#3. get all vhdr files (from a subject or from all BIDS_path)
vhdr_files=IO.get_all_vhdr_files(settings['BIDS_path'])

pp = PdfPages('Baseline_correction_output.pdf', keep_empty=False)

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
    used_channels = IO.read_M1_channel_specs(vhdr_file[:-9])

    #used_channels = IO.read_used_channels() #old
    print(used_channels)
    
    # extract used channels/labels from brainvision file, split up in cortex/subcortex/labels
    #dat_ is a dict
    dat_ = IO.get_dat_cortex_subcortex(bv_raw, ch_names, used_channels)
    dat_MOV=dat_['dat_label']
    seglengths = settings['seglengths']

    new_num_data_points = int((bv_raw.shape[1]/sf)*settings['resamplingrate'])
    offset_start = int((sf/seglengths[0]) / (sf/settings['resamplingrate']))
    y=np.empty((len(dat_MOV),new_num_data_points-offset_start), dtype=object)

    for m in range(len(dat_MOV)):
        
        T=len(dat_MOV[m])/sf #total time in sec
        if subject == '016':
            Df=55
            target_channel_corrected, onoff, raw_target_channel=offline_analysis.baseline_correction(y=-dat_MOV[m], Decimate=Df,method='baseline_rope', param=1e-1, thr=2e-1, normalize=True)
        else:
            Df=40

            target_channel_corrected, onoff, raw_target_channel=offline_analysis.baseline_correction(y=dat_MOV[m], Decimate=Df,method='baseline_rope', param=1e5, thr=2e-1, normalize=True)

        events=offline_analysis.create_events_array(onoff=onoff, raw_target_channel=dat_MOV[m], sf=sf)
        
        label=offline_analysis.generate_continous_label_array(L=new_num_data_points, sf=settings['resamplingrate'], events=events) 
        y[m]=label[offset_start:]    
       
        
        labels_plot=['original', 'clean','onset']



        t= np.arange(0.0, T , 1/sf)

   
        t2= np.arange(0.0, T, Df/sf)
        plt.figure(f+1, figsize=(8, 5))

        plt.subplot(1,2,m+1)
        plt.plot(t2[:500],raw_target_channel[:500], label=labels_plot[0])
        plt.plot(t2[:500],target_channel_corrected[:500], label=labels_plot[1])
        plt.plot(t2[:500],onoff[:500], label=labels_plot[2])
        plt.legend()
        plt.title('subject_'+ subject+ '_session' +sess+ '_run'+ run)
        
        if m==len(dat_MOV)-1: plt.savefig(pp, format='pdf')

        
        plt.figure(f+2, figsize=(8, 5))

        plt.subplot(2,1,m+1)
        plt.plot(t2,raw_target_channel, label=labels_plot[0])
        plt.plot(t2,target_channel_corrected, label=labels_plot[1])
        plt.plot(t2,onoff, label=labels_plot[2])
        plt.legend()
        plt.title('subject_'+ subject+ '_session' +sess+ '_run'+ run)
        
        if m==len(dat_MOV)-1: plt.savefig(pp, format='pdf')

        
#%%        
# for m in range(len(dat_MOV)):
#     plt.figure(f+3, figsize=(8, 4))
#     plt.subplot()
#     plt.plot(y[m])
#     plt.legend()
#     plt.title('LABELS: subject_'+ subject+ '_session' +sess+ '_run'+ run)

#     plt.savefig(pp, format='pdf')

pp.close()


               
    