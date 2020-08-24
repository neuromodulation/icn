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


from scipy import stats, signal
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
settings['out_path_process'] = "/mnt/Datos/BML_CNCRS/Spoc/ECoG_STN/SPOC_predictions_space/"


settings['frequencyranges']=[[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]]
settings['seglengths']=[1, 2, 2, 3, 3, 3, 10, 10, 10]
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
def NormalizeData(data):
    minv=np.min(data)
    maxv=np.max(data)
    data_new=(data - np.min(data)) / (np.max(data) - np.min(data))
    return data_new, minv, maxv

def DeNormalizeData(data,minv, maxv):
   
    data_new=(data + minv) * (maxv - minv)
    return data_new
#%%
laterality=["CON", "IPS"]
signals=["ECOG","STN"]

#clf=LinearRegression(normalize=True, n_jobs=-1)
# clf=LinearRegression()
# clf=ElasticNetCV(normalize=True, n_jobs=-1, tol=1e-2, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])

cv = KFold(n_splits=3, shuffle=False)
#%% CV split
len(settings['num_patients'])
for m, eeg in enumerate(signals):    

    for s in range(9,11):
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
                file_name='ECOG_epochs_sub_' + settings['num_patients'][s] + '_sess_'+subfolder[ss][4:]
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
            
            X=np.concatenate(X, axis=0)
            Y_con=np.concatenate(Y_con, axis=0)
            Y_ips=np.concatenate(Y_ips, axis=0)  
    
            nt,nc,ns,nf=np.shape(X)   

            #X in full beta
            x=np.squeeze(X[:,:,:,7])
    
           
            spoc= SPoC(n_components=1, log=None, reg='oas', transform_into ='csp_space', rank='full')
            
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
               
                    # #without ssd
                    gtr=np.squeeze(spoc.fit_transform(Xtr, Ztr))
                    gte=np.squeeze(spoc.transform(Xte))
                        
    
                    # gte=gtr.mean(axis=1)

                    gtr=gtr.var(axis=1)
                    gtr, Nan, Nan=NormalizeData(gtr)
                    signo=np.sign(np.dot(gtr,Ztr))
                    gte=gte.var(axis=1)
                    # gte=NormalizeData(gte)-NormalizeData(gte).mean()
                    # gte, mini, maxi=NormalizeData(signal.detrend(gte))
                    gte_n, mini, maxi=NormalizeData(gte)

                    # gte_n=signo*gte_n
                    # lala=DeNormalizeData(gte,mini,maxi)
                    # gte=stats.zscore(gte)
                    # Zte, mini, maxi=NormalizeData(Zte)

                    # fig, ax = plt.subplots()
                    # ax.plot(gte_n)
                    # ax.plot(Zte)
                                            
                    filters=spoc.filters_
                    patterns=spoc.patterns_
                    
                    lags, c=xcorr(gte,Zte, normed=True, detrend=False, maxlags=20)
                    maxi=np.argmax(c)                       
                            
                   
                    Ypre_te[mov].append(gte)
                    
                    r2_te=r2_score(Zte, gte_n)
                    if r2_te < 0: r2_te = 0
                    score_te[mov].append(r2_te)
                    lag[mov].append(lags[maxi])
                    corr[mov].append(c[maxi])

                          
                    Filters[mov].append(filters)
                    Patterns[mov].append(patterns)
                    
                    Label_te[mov].append(Zte)
                    
                    
                        
        
                        
            
           
            #%% save 
            predict_ = {
                "y_pred_test": Ypre_te,
                "y_test": Label_te,
                "score_te": score_te,
                "filters": Filters,
                "patterns": Patterns,
                "methods": spoc,
                "lag":  lag,
                "xcorr": corr
                
            }
            
            out_path_file = os.path.join(settings['out_path_process']+ settings['num_patients'][s]+'predictions_'+eeg+'_space'+'_'+str(subfolder[ss])+'.npy')
            np.save(out_path_file, predict_)        
            
            gc.collect()
            
                
        # %% Plot the True mov and the predicted
            fig, ax = plt.subplots(1, 1, figsize=[10, 4])
            ind_best=np.argmax(corr['CON'])
            Ypre_te_best=NormalizeData(Ypre_te['CON'][ind_best])[0]
            label_test_best=Label_te['CON'][ind_best]
            lags, c=xcorr(Ypre_te_best,label_test_best, normed=True, detrend=False, maxlags=None)
                    
            maxi=np.argmax(c)
            #times = raw.times[meg_epochs.events[:, 0] - raw.first_samp]
            ax.plot(Ypre_te_best, color='b', label='Predicted mov')
            ax.plot(label_test_best, color='r', label='True mov')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Movement')
            ax.set_title('SPoC mov Predictions')
            ax.text(0.33, 0.9, 'R2={:0.02f}'.format(score_te['CON'][ind_best]),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,fontsize=12) 
            ax.text(0.33, 0.85, 'xcorr={:0.02f}'.format(c[maxi]),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,fontsize=12) 
            ax.text(0.33, 0.8, 'lag={:0.02f}'.format(lags[maxi]),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,fontsize=12) 
            fig.suptitle(eeg+'-Subject_'+ settings['num_patients'][s], fontsize=14, fontweight='bold')
            plt.legend()
            plt.show()
            
        # fig, ax = plt.subplots(1, 1, figsize=[10, 4])
        # # ind_best=np.argmax(score_te['IPS'])
        # Ypre_te_best=Ypre_te['IPS'][ind_best]
        # label_test_best=Label_te['IPS'][ind_best]
        # lags, c=xcorr(Ypre_te_best,label_test_best, normed=True, detrend=False, maxlags=None)
                
        # maxi=np.argmax(c)
        # #times = raw.times[meg_epochs.events[:, 0] - raw.first_samp]
        # ax.plot(Ypre_te_best, color='b', label='Predicted mov')
        # ax.plot(label_test_best, color='r', label='True mov')
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Movement')
        # ax.set_title('SPoC mov Predictions')
        # ax.text(0.33, 0.9, 'R2={:0.02f}'.format(score_te['CON'][ind_best]),
        # verticalalignment='bottom', horizontalalignment='right',
        # transform=ax.transAxes,fontsize=12) 
        # ax.text(0.33, 0.85, 'xcorr={:0.02f}'.format(c[maxi]),
        # verticalalignment='bottom', horizontalalignment='right',
        # transform=ax.transAxes,fontsize=12) 
        # ax.text(0.33, 0.8, 'lag={:0.02f}'.format(lags[maxi]),
        # verticalalignment='bottom', horizontalalignment='right',
        # transform=ax.transAxes,fontsize=12) 
        # fig.suptitle(eeg+'-Subject_'+ settings['num_patients'][s], fontsize=14, fontweight='bold')
        # plt.legend()
        # plt.show()
            
            
