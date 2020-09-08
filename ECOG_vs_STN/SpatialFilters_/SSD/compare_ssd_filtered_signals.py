#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:30:37 2020

@author: victoria
"""

#%%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
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
from myssd import SSD

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet


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

from FilterBank import *
# plt.close('all')

#%%
settings = {}

settings['BIDS_path'] = "/mnt/Datos/BML_CNCRS/Data_BIDS/"
settings['out_path'] = "/mnt/Datos/BML_CNCRS/Spoc/"
settings['out_path_process'] = "/mnt/Datos/BML_CNCRS/Spoc/ECoG_STN/"


settings['frequencyranges']=[[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]]
settings['seglengths']=[1, 2, 2, 3, 3, 3, 10, 10, 10]
# settings['num_patients']=['000', '004', '005', '007', '008', '009', '010', '013', '014']
settings['num_patients']=['000', '001', '004', '005', '006', '007', '008', '009', '010', '013', '014']


settings['BIDS_path']=settings['BIDS_path'].replace("\\", "/")
settings['out_path']=settings['out_path'].replace("\\", "/")

#%%

def DetecBadTrials(X,y, verbose=True):
    #based on eq.16 of the paper "Dimensionality reduction for the analysis of brain oscillations"
    #calcule single global variance value (GVV)
    nt, nc, ns= np.shape(X)
    TrialVar=np.zeros((nt,nc,1))

    for c in range(nc):
        for n in range(nt):
            TrialVar[n,c,:]=np.var(X[n,c,:])
    
    GVV=np.squeeze(np.mean(TrialVar, axis=1))
    Q5=np.percentile(GVV, 5)
    Q95=np.percentile(GVV, 95)

    Thr=Q95+3*(Q95-Q5)
    
    #elimiate trails with large variance
    index_good=np.where(GVV<Thr)[0]
    index_bad=np.where(GVV>Thr)[0]

    if verbose:
        if len(index_bad)>0 :
            print('Detected bad trials')
    
    X_clean=X[index_good,:,:]
    Y_clean=y[index_good]
    
    return X_clean,Y_clean
    
        
            
    
#%%
# plt.close("all")
laterality=["CON"]
signal=["ECOG", "STN"]

filtering=["YES", "NO"]

cv = KFold(n_splits=3, shuffle=False)
#%% CV split
len(settings['num_patients'])
for m, eeg in enumerate(signal):
    fig, axs = plt.subplots(2,8)
    for f, fil in enumerate(filtering):
        
        for s in range(1):
            gc.collect()
        
            subject_path=settings['BIDS_path'] + 'sub-' + settings['num_patients'][s]
            subfolder=IO.get_subfolders(subject_path)
           
            for ss in range(1):
                X=[] #to append data
                Y_con=[]
                Y_ips=[]
                      
        
                print('RUNNIN SUBJECT_'+ settings['num_patients'][s]+ '_SESS_'+ str(subfolder[ss]) + '_SIGNAL_' + eeg)
        
                list_of_files = os.listdir(settings['out_path']) #list of files in the current directory
                if fil == "YES":
                    file_name=eeg+'_epochs_sub_' + settings['num_patients'][s] + '_sess_'+subfolder[ss][4:]
                else:                           
                    file_name=eeg+'_epochs_wofb_sub_' + settings['num_patients'][s] + '_sess_'+subfolder[ss][4:]

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
                
                X=np.concatenate(X, axis=0)
                Y_con=np.concatenate(Y_con, axis=0)
                Y_ips=np.concatenate(Y_ips, axis=0)  
                
                                    
                
        
                for l, mov in enumerate(laterality):
                    print("training %s" %mov)
                    
                    if l==0:
                        label=Y_con
                    else:
                        label=Y_ips
                    
                    
                    if fil=="NO":
                        x=np.squeeze(X)
                        x, label= DetecBadTrials(x,label)
                    nfb, nt,nc,ns=np.shape(X)   
                              
                    result_lm=[]
                    result_rm=[]
                    
                    label_test=[]
                    label_train=[]
                    
                    onoff_test=[]
                    onoff_train=[]
                    
                
    
                    
                        
                    for fb in range(len( settings['frequencyranges'])): 
                       
                        if fil=="YES":
                            x=np.squeeze(X[fb])
                            x, label= DetecBadTrials(x,label)
                            
                    
                        Xtr=x
                        Xtr=Xtr.astype('float64')
                        if fil=="NO":        
                            band= settings['frequencyranges'][fb]
                            freq = [band, band+np.array([-2,2]), band+np.array([-1,1])]
                            ssd=SSD(n_components=nc-1,freq=freq, sampling_freq=1000.0, reg='oas',rank='full')
                            
                           
                            
                            Xtr=ssd.fit_transform(Xtr)
                                                
                        
                            # plt.subplot(111)
                            axs[f,fb].psd(Xtr[0,0,:],Fs=1000,label='denoised')

                        else:
                            # plt.subplot(112)
                            axs[f,fb].psd(Xtr[0,0,:],Fs=1000,label='original')
                    plt.suptitle('without SSD')
                    plt.subplots_adjust(hspace=0.5)

                    plt.figtext(0.5, 0.5, 'with SSD', ha='center', va='center')


                    
                           


            
