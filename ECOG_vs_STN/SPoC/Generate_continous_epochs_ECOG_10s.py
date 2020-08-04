# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:49:39 2020

@author: VPeterson
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/victoria/icn/icn_m1')
import filter
import IO
import online_analysis
import offline_analysis
import rereference
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
from mne.decoding import CSP
from mne import Epochs
from mne.decoding import SPoC
mne.set_log_level(verbose='warning') #to avoid info at terminal

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import gc

#%%

def epoch_data_ML(data, events, sf, toffset=0.1, twindows=1):
    """
    this function segments data in rest and movement epochs.
    Rest epoch are taken between trials from the tmin sec before target onset and tmax sec
    after target onset

    Parameters
    ----------
    data : array, shape(n_channels, n_samples)
        either cortex of subcortex data to be epoched.
    events : array, shape(n_events,2)
        All events that were found by the function
        'create_events_array'. 
        The first column contains the event time in samples and the second column contains the event id.
    sf : int, float
        sampling frequency of the raw_data.
    tmin : float
        Start time before event (in sec). 
        If nothing is provided, defaults to 1.
    tmax : float
        Stop time after  event (in sec). 
        If nothing is provided, defaults to 1.
    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    X : array, shape(n_events, n_channels, n_samples)
        epoched data
    Y : array, shape(n_events, n_samples)
        sample-wise label information of data.
.

    """
    
    
    #get time_events index    
    mask_start=events[:,1]==1
    
    n_trials=len(events)
    #labels
            
    Y=np.zeros(n_trials)
    Y[mask_start]=1
    
    #time up
    time_up=[x - events[i - 1] for i, x in enumerate(events[:,0])][1:]
    time_up=np.asarray(time_up)[:,0]
    min_time_up=min(time_up)
    #check inputs
    if twindows > min_time_up:
        Warning('twindows too large. It should be lower than={:3.2f}'.format(min_time_up))
        twindows=min_time_up
      
        
    for i in range(len(events)):
        start_epoch=int(np.round((events[i,0]+toffset)*sf))
        stop_epoch=int(np.round((events[i,0]+toffset+twindows)*sf))
        
        if stop_epoch > data.shape[1]:
            Warning('Not enough data to extract last trial given the desided window length')
            Y[:-1]
        else:
            epoch=data[:,start_epoch:stop_epoch]
        #reshape data (n_events, n_channels, n_samples)
        nc, ns=np.shape(epoch)
        epoch=np.reshape(epoch,(1, nc,ns))
        if i==0:
            X=epoch
        else:
            X=np.vstack((X,epoch))
           
    return X, Y

def continous_epoch_data(data, sf, t_min=0.0, t_windows=1, t_stride=0.04):
    
    
    time_start_epoch = np.arange(0, data.shape[1], t_stride)
    for e in range(len(time_start_epoch)):
        start_epoch=int(np.round((time_start_epoch[e]+t_min)*sf))
        stop_epoch=start_epoch + int(np.round(t_windows*sf))
        
        
        epoch=data[:,start_epoch:stop_epoch]
        
        if epoch.shape[1]<int(round(t_windows*sf)):
            break
        #reshape data (n_events, n_channels, n_samples)
        nc, ns=np.shape(epoch)
        epoch=np.reshape(epoch,(1, nc,ns))
        if e==0:
            X=epoch
        else:
            X=np.vstack((X,epoch))
    return X


def t_f_transform(x, sample_rate, f_ranges, line_noise):
    """
    calculate time frequency transform with mne filter function
    """
    filtered_x = []

    for f_range in f_ranges:
        if line_noise in np.arange(f_range[0], f_range[1], 1):
            #do line noise filtering

            x = mne.filter.notch_filter(x=x, Fs=sample_rate, 
                freqs=np.arange(line_noise, 4*line_noise, line_noise), 
                fir_design='firwin', verbose=False, notch_widths=2)

        h = mne.filter.create_filter(x, sample_rate, l_freq=f_range[0], h_freq=f_range[1], \
                                     fir_design='firwin', verbose=False, l_trans_bandwidth=2, h_trans_bandwidth=2)
        filtered_x.append(np.convolve(h, x, mode='same'))
 
    return np.array(filtered_x)


def transform_channels(bv_raw, settings, sample_rate, line_noise):
    """
    calculate t-f-transform for every channel
    :param bv_raw: Raw (channel x time) datastream
    :return: t-f transformed array in shape (len(f_ranges), channels, time)
    """
    x_filtered = np.zeros([len(settings['frequencyranges']), bv_raw.shape[0], bv_raw.shape[1]])
    for ch in range(bv_raw.shape[0]):
        x_filtered[:, ch, :] = t_f_transform(bv_raw[ch, :], sample_rate, settings['frequencyranges'], line_noise)
    return x_filtered

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
settings['max_dist_cortex']=20
settings['max_dist_subcortex']=5
settings['normalization_time']=10
settings['frequencyranges']=[[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]]
settings['seglengths']=[1, 2, 2, 3, 3, 3, 10, 10, 10]
settings['num_patients']=['000', '001', '004', '005', '006', '007', '008', '009', '010', '013', '014']

settings['BIDS_path']=settings['BIDS_path'].replace("\\", "/")
settings['out_path']=settings['out_path'].replace("\\", "/")

with open('settings/mysettings.json', 'w') as fp:
    json.dump(settings, fp)
    
settings = IO.read_settings('mysettings')
#2. write _channels_MI file
IO.write_all_M1_channel_files()

#%%
len(settings['num_patients'])
for s in range(len(settings['num_patients'])
):
    
    

    subject_path=settings['BIDS_path'] + 'sub-' + settings['num_patients'][s]
    subfolder=IO.get_subfolders(subject_path)

    for ss in range(len(subfolder)):
        
        X=[] #to append data
        Y_con=[]
        OnOff_con=[]
        Y_ips=[]
        OnOff_ips=[]
    
        
           
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
    
            
             # #%% REREFERENCE
            dat_ECOG, dat_STN =rereference.rereference(run_string=vhdr_file[:-10], data_cortex=dat_ECOG, data_subcortex=dat_STN)

        
        
            #%% filter data
            line_noise = IO.read_line_noise(settings['BIDS_path'],subject)
            x_filtered=transform_channels(dat_ECOG, settings, sf, line_noise)
            
            #%% create MNE object
            # Build epochs as sliding windows over the continuous raw file
            channels_ecog=[ch_names[i] for i in ind_cortex] 
            info_ecog = mne.create_info(ch_names=channels_ecog, sfreq=sf, ch_types='ecog')           
               
            
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
  
    #%% epoch data with mne
        
            f_ranges=settings['frequencyranges']
            data=[]

            for fb in range(len(f_ranges)): 
                raw_ecog = mne.io.RawArray(x_filtered[fb], info_ecog)
                            
                    
                events_ecog=mne.make_fixed_length_events(raw_ecog, id=1, start=0, stop=None, duration=.1)
                ecog_epoch=Epochs(raw_ecog, events_ecog, event_id=1, tmin=0, tmax=1, baseline=None)
                aux_data=ecog_epoch.get_data().astype('float32')
                data.append(aux_data)
                    
               # print(np.shape(data))
            gc.collect()

            label_con=y_con[:np.shape(data)[1]]
            label_ips=y_ips[:np.shape(data)[1]]
            
            onoff_con=onoff_mov_con[:np.shape(data)[1]]
            onoff_ips=onoff_mov_ips[:np.shape(data)[1]]
            
            # X.append(data)
            # Y_con.append(label_con)
            # OnOff_con.append(onoff_con)
            # Y_ips.append(label_ips)
            # OnOff_ips.append(onoff_ips)
        

            # X=np.concatenate(data, axis=1)
            # Y_con=np.concatenate(label_con, axis=0)
            # Y_ips=np.concatenate(label_ips, axis=0)            
            # OnOff_con=np.concatenate(onoff_con, axis=0)
            # OnOff_ips=np.concatenate(onoff_ips, axis=0)
            #%% save
            sub_ = {
                "ch_names" : ch_names, 
                "subject" : subject, 
                "used_channels" : used_channels, 
                "fs" : sf, 
                "line_noise" : line_noise, 
                "label_con" : label_con, 
                "label_ips" : label_ips,         
                "onoff_con" : onoff_con, 
                "onoff_ips" : onoff_ips, 
                "epochs" : data,
                "session": sess,
                "run": run,
                "info_mne" : info_ecog,
              
                
            }
            
            out_path = os.path.join(settings['out_path'],'epochs_10s_sub_' + subject +'_sess_' +sess + '_run_'+ run + '.p')
            
            with open(out_path, 'wb') as handle:
                pickle.dump(sub_, handle, protocol=pickle.HIGHEST_PROTOCOL)   
        
# channels_mov=[ch_names[i] for i in ind_label] 
            # info_mov = mne.create_info(ch_names=channels_mov, sfreq=25, ch_types='stim')
            # f_ranges=settings['frequencyranges']
            #%%
            # for fb in range(len(f_ranges)): 
                
               

            
                
            #     #events_mov=mne.make_fixed_length_events(raw_mov, id=1, start=0, stop=None, duration=.04)
            #     #mov_epoch=Epochs(raw_mov, events_mov, event_id=1, tmin=0, tmax=1, baseline=None)
                # data = ecog_epoch.get_data()
            #     if f==0:
            #         X[fb]=data
            #     else:
            #         X[fb]=np.vstack((X[fb],data))
            
            # I need this for making the right epoching.        
            #ind=events_ecog[:,0]+1000<dat_MOV.shape[-1]
                
        
                
            #mov=np.zeros((dat_MOV.shape[0], round(len(dat_MOV[1])/40)+1))
            #mov=np.zeros((dat_MOV.shape[0], data.shape[0]))   

    # X.append(x_filtered)
        # Y_con.append(y_con)
        # OnOff_con.append(onoff_mov_con)
        # Y_ips.append(y_ips)
        # OnOff_ips.append(onoff_mov_ips)
    
    # f_ranges=settings['frequencyranges']
    # data=[]
    # for fb in range(len(f_ranges)): 
    #     raw_ecog = mne.io.RawArray(x_filtered[fb], info_ecog)
                    
            
    #     events_ecog=mne.make_fixed_length_events(raw_ecog, id=1, start=0, stop=None, duration=.04)
    #     ecog_epoch=Epochs(raw_ecog, events_ecog, event_id=1, tmin=0, tmax=1, baseline=None)
     
    #     data.append(ecog_epoch.get_data())    
    # X=np.concatenate(X, axis=2)
    # Y_con=np.concatenate(Y_con, axis=0)
    # Y_ips=np.concatenate(Y_ips, axis=0)            
    # OnOff_con=np.concatenate(OnOff_con, axis=0)
    # OnOff_ips=np.concatenate(OnOff_ips, axis=0)
        
        # if f==0:
        #     X=x_filtered
        #     Y_con=y_con
        #     OnOff_con=onoff_mov_con
        #     Y_ips=y_ips
        #     OnOff_ips=onoff_mov_ips
        # else:
        #     X=np.concatenate((X,x_filtered), axis=2)
        #     Y_con=np.concatenate((Y_con,y_con), axis=0)
        #     Y_ips=np.concatenate((Y_ips,y_ips), axis=0)            
        #     OnOff_con=np.concatenate((OnOff_con,onoff_mov_con), axis=0)
        #     OnOff_ips=np.concatenate((OnOff_ips,onoff_mov_ips), axis=0)
        # print(X.shape)
        # print(Y_con.shape)    
    # classi_result = {
    #     "subject" : subject, 
    #     "lm_ypre" :Ypre_, 
    #     "rd_ypre": 
        
    # }
    
    # out_path = os.path.join(settings['out_path'],'sub_' + subject + '.p')
    
    # with open(out_path, 'wb') as handle:
    #     pickle.dump(sub_, handle, protocol=pickle.HIGHEST_PROTOCOL)       
    # print(np.mean(result_lm))
    # print(np.mean(result_rm))

                



            
    #     # Classification pipeline with SPoC spatial filtering and Ridge Regression
    #     clf = make_pipeline(spoc, Ridge())
        
        
    #     # Run cross validaton
    #     y_preds = cross_val_predict(clf, X, Y, cv=cv)
        
    


    #         # #%% filter data
    #         # nt,nc,ns=X.shape
    #         # X_filtered=np.zeros((nt,nc,ns, len(f_ranges)))
        
    #         # for f, f_range in enumerate(f_ranges):
    #         #     X_filtered[:,:,:,f]=mne.filter.filter_data(X, sf, f_range[0], f_range[1])
    #         # #%% csp
            
            
    #     # csp = CSP(n_components=2, reg='empirical', log=True, norm_trace=False, cov_est='epoch')
    #     # #learn csp filters for each FB
    #     # Gtr=np.zeros((nt,2*len(f_ranges)))
    #     # for f in range(len(f_ranges)):
    #     #     Gtr[:,f*2:f*2+2]=csp.fit_transform(X_filtered[:,:,:,f],Y)
        
        
    
            
    #     # filter_len=501
    #     # 
    #     # filter_fun = np.zeros([len(f_ranges), filter_len])
    
    #     # for a, f_range in enumerate(f_ranges):
    #     #     h = mne.filter.create_filter(None, sf, l_freq=f_range[0], h_freq=f_range[1], 
    #     #                         fir_design='firwin', filter_length='500ms', l_trans_bandwidth=3.5, h_trans_bandwidth=3.5)
    
    #     #     filter_fun[a, :] = h
    # #     #%%
    # # #filter_fun = filter.calc_band_filters(, sample_rate=sf, filter_len=501)
    # # filtered = np.zeros((filter_fun.shape[0],dat_notch_filtered.shape[0]+1))
    # # for filt in range(filter_fun.shape[0]):
    # #     filtered=scipy.signal.convolve(filter_fun[filt,:], 
    # #                                             # dat_notch_filtered, mode='same')