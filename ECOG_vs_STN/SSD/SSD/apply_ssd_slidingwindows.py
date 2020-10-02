# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:49:39 2020

@author: VPeterson
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/victoria/icn/icn_m1')
import IO
import preprocessing
import numpy as np
import json
import os
import pickle 
import mne
from matplotlib import pyplot as plt
import gc
sys.path.insert(1, '../Utilities')
from ssd import  SSD
from mne.utils import _time_mask
from matplotlib.backends.backend_pdf import PdfPages

mne.set_log_level(verbose='warning') #to avoid info at terminal
plt.close("all")

pp = PdfPages('SSD_spectrum_analysis.pdf', keep_empty=False)

#%%
def freq_mask(freqs, fmin, fmax):
    """convenience function to select frequencies"""
    return _time_mask(freqs, fmin, fmax)
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
settings['frequencyranges']=[[4, 250]]
settings['num_patients']=['000', '001', '004', '005', '006', '007', '008', '009', '010', '013', '014']
settings['seglengths']=[1, 2, 2, 3, 3, 3, 10, 10, 10]

settings['BIDS_path']=settings['BIDS_path'].replace("\\", "/")
settings['out_path']=settings['out_path'].replace("\\", "/")

#%%
signal=["STN", "ECOG"]
#%%
for m, eeg in enumerate(signal):    
    
    
    
    for s in range(len(settings['num_patients'])):
        
        
    
        subject_path=settings['BIDS_path'] + 'sub-' + settings['num_patients'][s]
        subfolder=IO.get_subfolders(subject_path)
        
        for ss in range(len(subfolder)):
            X=[] #to append data
            Y_con=[]
            Y_ips=[]    
               
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
                dat_ECOG, dat_STN =preprocessing.rereference(run_string=vhdr_file[:-10], bv_raw=bv_raw, get_ch_names=False, get_cortex_subcortex=True)
                if eeg == "STN" and dat_STN is None:
                    continue
                
                #%% GET TARGET
                recording_length = bv_raw.shape[1] 
                line_noise = IO.read_line_noise(settings['BIDS_path'],subject)
                new_num_data_points = int((recording_length/sf)*settings['resamplingrate'])         
                   
                
                mov_ch=int(len(dat_MOV)/2)
                con_true = np.empty(mov_ch, dtype=object)
                wl=int(recording_length/(new_num_data_points))
                onoff=np.zeros(np.size(dat_MOV[0]))
    
                for m in range(mov_ch):
                    target_channel_corrected=dat_MOV[m+mov_ch]
    
                   
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
      
           
                #     #%% APPEND
                if eeg== "ECOG":
                    X.append(dat_ECOG)
                else:
                    X.append(dat_STN)
                Y_con.append(y_con)
                Y_ips.append(y_ips)
           
            gc.collect()
            if eeg == "STN" and dat_STN is None:
                continue    
            X=np.concatenate(X, axis=1)
            Y_con=np.concatenate(Y_con, axis=0)
            Y_ips=np.concatenate(Y_ips, axis=0)
            
            #%% create mne object
            used_channels = IO.read_M1_channel_specs(vhdr_file[:-10])
            if eeg == "STN" and dat_STN is None:
                    continue
            if eeg == "ECOG":
                if used_channels["subcortex"] is None:
                    picks=used_channels["cortex"]
                else:
                    picks=used_channels["cortex"]-len(used_channels["subcortex"])
            else:
                 picks=used_channels["subcortex"]
            channels_ecog=[ch_names[i] for i in picks] 
            info_ecog = mne.create_info(ch_names=channels_ecog, sfreq=sf, ch_types='ecog')  
            raw= mne.io.RawArray(X, info_ecog)   
            
            #%% plot psd
            # raw.plot_psd(fmax=250,show=False, average=True)
            
            nfft=2048
            #%% apply ssd in slidding windows
            delta1=2
            delta2=2
            PSD=[]
            for f in range(2,250):
                print(f)
                freqs_sig = f, f + delta1
                freqs_noise = f-delta2, f + delta1+delta2
                ssd = SSD(filt_params_signal=dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                                      l_trans_bandwidth=1, h_trans_bandwidth=1,
                                      fir_design='firwin'),\
                      filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                      l_trans_bandwidth=1, h_trans_bandwidth=1,
                                      fir_design='firwin'), 
                      sampling_freq=sf, picks=picks, rank="full", n_fft=nfft)    
                    
                #fit and transform
                # ssd.fit(raw)
                ssd.fit(raw.copy().crop(0, 120))

                #%%
                ssd_sources = ssd.transform(raw)
            
                #%%
                spec_ratio = ssd.spec_ratio
                sorter = ssd.sorter_spec
                eigenvals=ssd.eigvals_
                
                aux_diff_mean=-np.diff(eigenvals)[:2].mean()
                # aux_diff_mean=eigenvals[0]- eigenvals[-1]
                # aux_diff_mean=spec_ratio[0]- spec_ratio[-1]
    
                PSD.append(aux_diff_mean)
                
                # n_com=np.shape(np.where(spec_ratio[sorter]>1))[1]
                # n_com=2
                #plot spectral ratio (see Eq. 24 in Nikulin 2011)
                # plt.figure()
                # plt.plot(spec_ratio, color='black')
                # plt.plot(spec_ratio[sorter], color='orange', label='sorted eigenvalues')
                # plt.xlabel("Eigenvalue Index")
                # plt.ylabel(r"Spectral Ratio $\frac{P_f}{P_{sf}}$")
                # plt.legend()
                # plt.axhline(1, linestyle='--')
                
                # psd, freqs = mne.time_frequency.psd_array_welch(
                #     ssd_sources, sfreq=raw.info['sfreq'],  n_fft=nfft)
                
                # below50 = freq_mask(freqs, 0, 50)
                # bandfilt = freq_mask(freqs, freqs_sig[0],freqs_sig[1])
                
                # plt.figure()
                # plt.loglog(freqs[below50], psd[0, below50], label='max SNR')
                # plt.loglog(freqs[below50], psd[-1, below50], label='min SNR')
                # plt.loglog(freqs[below50], psd[:, below50].mean(axis=0), label='mean')
                # plt.fill_between(freqs[bandfilt], 0, 100,\
                #                 color='green', alpha=0.5)
                # plt.xlabel("log(frequency)")
                # plt.ylabel("log(power)")
                # plt.legend()
               
                    
                    
                   
                    # PSD=np.concatenate(PSD,axis=0)    
            
            
            psd_raw, freqs = mne.time_frequency.psd_array_welch(
                    dat_ECOG, sfreq=raw.info['sfreq'], n_fft=nfft, fmax=250)
                       
            plt.figure(figsize=[18, 10])
            plt.subplot(121)
            plt.semilogy(freqs,psd_raw.mean(axis=0), label='raw')
            plt.xlabel("frequency")
            plt.ylabel("log(power)")
            plt.legend()
            plt.suptitle(eeg+'-S'+settings['num_patients'][s]+ '-' +subfolder[ss])
            plt.grid(True)
            plt.title('Average spectrum', fontsize=12)
           
            
            plt.subplot(122)
            plt.semilogy(PSD)
            plt.xlabel("frequency")
            plt.ylabel("log(mean of diff.)")
            # plt.title(eeg+'-S'+settings['num_patients'][s]+ '-' +subfolder[ss])
            plt.grid(True)
            plt.title('Mean of differences between the consecutive 2 largest eigenvalues', fontsize=12)
            plt.savefig(pp, bbox_inches='tight',format='pdf')
pp.close()

            # #%% save
            # sub_ = {
            #     "ch_names" : ch_names, 
            #     "subject" : subject, 
            #     "used_channels" : used_channels, 
            #     "fs" : sf, 
            #     "line_noise" : line_noise, 
            #     "label_con" : y_con, 
            #     "label_ips" : y_ips,         
            #     "onoff_con" : onoff_mov_con, 
            #     "onoff_ips" : onoff_mov_ips, 
            #     "epochs" : data,
            #     "session": sess,
            #     "run": run,
              
                
            # }
            
            # out_path = os.path.join(settings['out_path'],'ECOG_epochs_sub_' + subject +'_sess_' +sess + '_run_'+ run + '.p')
            
            # with open(out_path, 'wb') as handle:
            #     pickle.dump(sub_, handle, protocol=pickle.HIGHEST_PROTOCOL)   
