#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:00:39 2020

@author: victoria
"""


import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import itertools
import mne
from mne.decoding import CSP
from mne import Epochs
from mne.decoding import SPoC
mne.set_log_level(verbose='warning') #to avoid info at terminal
import pickle 
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/victoria/icn/icn_m1')
import IO
import os


from scipy import stats
from collections import OrderedDict
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from bayes_opt import BayesianOptimization
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
import gc
from sklearn.preprocessing import StandardScaler
from myssd import SSD
plt.close('all')

#%%
settings = {}

settings['BIDS_path'] = "/mnt/Datos/BML_CNCRS/Data_BIDS/"
settings['out_path'] = "/mnt/Datos/BML_CNCRS/Spoc/"
settings['out_path_process'] = "/mnt/Datos/BML_CNCRS/Spoc/ECoG_STN/"


settings['frequencyranges']=[[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]]
# settings['num_patients']=['000', '004', '005', '007', '008', '009', '010', '013', '014']
settings['num_patients']=['000', '001', '004', '005', '006', '007', '008', '009', '010', '013', '014']


settings['BIDS_path']=settings['BIDS_path'].replace("\\", "/")
settings['out_path']=settings['out_path'].replace("\\", "/")
#%%
def xcorr(x, y, normed=True, detrend=False, maxlags=10):
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed=True
    # Returns inner products when normed=False
    # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)
    # Optional detrending e.g. mlab.detrend_mean

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')
    
    if detrend:
        import matplotlib.mlab as mlab
        x = mlab.detrend_mean(np.asarray(x)) # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))
    
    c = np.correlate(x, y, mode='full')

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    return lags, c

def psd_epochs (X):
    ne, nc, ns= X.shape()
    for e in range(ne):
        for c in range (nc):
            psds, freqs = mne.time_frequency.psd_welch(X[ne,nc,:], picks=['O1'], fmin=13.0, fmax=22.0)

    

#%%
laterality=["CON", "IPS"]
signal=["ECOG", "STN"]


cv = KFold(n_splits=3, shuffle=False)
#%% CV split
len(settings['num_patients'])
for m, eeg in enumerate(signal):    

    for s in range(1):
        gc.collect()
    
        subject_path=settings['BIDS_path'] + 'sub-' + settings['num_patients'][s]
        subfolder=IO.get_subfolders(subject_path)
       
        for ss in range(len(subfolder)):
            X=[] #to append data
            Y_con=[]
            Y_ips=[]
                  
    
            print('RUNNIN SUBJECT_'+ settings['num_patients'][s]+ '_SESS_'+ str(subfolder[ss]) + '_SIGNAL_' + eeg)
    
            list_of_files = os.listdir(settings['out_path']) #list of files in the current directory
            if eeg=="ECOG":
                file_name='epochs_sub_' + settings['num_patients'][s] + '_sess_'+subfolder[ss][4:]
            else:
                file_name='STN_epochs_sub_' + settings['num_patients'][s] + '_sess_'+subfolder[ss][4:]
            for each_file in list_of_files:
                
                if each_file.startswith(file_name):  #since its all type str you can simply use startswith
                           
                    with open(settings['out_path'] + each_file, 'rb') as handle:
                        sub_ = pickle.load(handle)    
           
                        data=sub_['epochs']
                        label_ips=sub_['label_ips']
                        label_con=sub_['label_con']
                                                              
                        X.append(data)
                        Y_con.append(label_con)
                        Y_ips.append(label_ips)
            
            gc.collect()
            
            X=np.concatenate(X, axis=1)
            Y_con=np.concatenate(Y_con, axis=0)
            Y_ips=np.concatenate(Y_ips, axis=0)  
    
            nf,nt,nc,ns=np.shape(X)   

            #X in full gamma
            x=np.squeeze(X[4])
            #SSD
            band= settings['frequencyranges'][3] 
            freq = [band, band+np.array([-2,2]), band+np.array([-1,1])]
    
                      
            Ypre_tr= OrderedDict()
            score_tr= OrderedDict()
            Ypre_te= OrderedDict()
            score_te= OrderedDict()
            lag= OrderedDict()
            corr= OrderedDict()

            Patterns= OrderedDict()
            Filters= OrderedDict()
            Label_tr= OrderedDict()
            Label_te= OrderedDict()
    
            for l, mov in enumerate(laterality):
                print("training %s" %mov)
                score_tr[mov] = []
                score_te[mov] = []
                lag[mov] = []
                corr[mov] = []
                Ypre_tr[mov] = []
                Ypre_te[mov] = []
                Label_tr[mov] = []
                Label_te[mov] = []
                Patterns[mov]=[]
                Filters[mov]=[]
                if l==0:
                    label=Y_con
                else:
                    label=Y_ips
                
                # #z-scored label also
                # label=stats.zscore(label)
                          
                result_lm=[]
                result_rm=[]
                
                label_test=[]
                label_train=[]
                
                onoff_test=[]
                onoff_train=[]
                
                
                for train_index, test_index in cv.split(label):
                    Ztr, Zte=label[train_index], label[test_index]
                    Xtr, Xte=x[train_index,:,:], x[test_index,:,:]
                    Xtr=Xtr.astype('float64')
                    Xte=Xte.astype('float64')
                    
                    for nn in range(1,nc+1):
                        ssd=SSD(n_components=nn,freq=freq, sampling_freq=1000,  reg='oas',rank='full')
    
     
                        #SSD
                        X_denoised=ssd.fit_transform(Xtr)
                        # W_ss=ssd.filters_
                        # A_ss=ssd.patterns_
                        
                        # W_ss = W_ss[:1]
                        # A_ss = A_ss[:,:1]
                        
                        # A_ssd=ssd.patterns_
                        # pick_patterns =A_ssd[:nn]
                        # X_denoised2 = np.asarray([np.dot(pick_patterns.T, epoch) for epoch in X_denoised])
                        
                        # X_denoised2 = np.asarray([np.dot(np.multiply(W_ss, A_ss.T), epoch) for epoch in Xtr])
                        # X_denoised = np.asarray([np.dot(A_ss, epoch) for epoch in X_denoised])
                       
                        plt.figure()
                        plt.psd(X_denoised[0,0,:],Fs=1000,label='denoised')
                        plt.psd(Xtr[0,0,:],Fs=1000, label='true')
                        # plt.psd(X_denoised2[0,0,:],Fs=1000, label='denoised2')
                        plt.legend()
                    
                        lala0=mne.time_frequency.psd_array_welch(Xtr, sfreq=1000, fmin=0, fmax=60)
                        # lala1=mne.time_frequency.psd_array_welch(X_denoised2, sfreq=1000, fmin=0, fmax=500)
                        lala2=mne.time_frequency.psd_array_welch(X_denoised, sfreq=1000, fmin=0, fmax=60)
    
                        plt.figure()
                        plt.plot(lala0[1], lala0[0][0,0,:],label='true')
                        # plt.plot(lala1[1], lala1[0][0,0,:],label='denoised2') 
                        plt.plot(lala2[1], lala2[0][0,0,:],label='denoised') 

                        plt.legend()
 