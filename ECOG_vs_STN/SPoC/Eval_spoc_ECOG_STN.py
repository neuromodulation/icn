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
reg=ElasticNet(max_iter=1000)
space_LM = [Real(0, 1, "uniform", name='alpha'),Real(0, 1, "uniform", name='l1_ratio')]
#%%
def optimize_enet(x,y):
    
    @use_named_args(space_LM)
    def objective(**params):
        reg.set_params(**params)
        cval = cross_val_score(reg, x, y, scoring='r2', cv=3)
        cval[np.where(cval < 0)[0]] = 0
    
        return -cval.mean()

    res_gp = gp_minimize(objective, space_LM, n_calls=20, random_state=0)
    return res_gp

# def enet_train(alpha,l1_ratio,x,y):
#     clf=ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000,normalize=False)
#     #clf.fit(x,y)
    
#     cval = cross_val_score(clf, x, y, scoring='r2', cv=3)
#     cval[np.where(cval < 0)[0]] = 0
#     return cval.mean()
    
#     return clf.score(x, y)
# def optimize_enet(x,y):
#     """Apply Bayesian Optimization to select enet parameters."""
#     def function(alpha, l1_ratio):
          
#         return enet_train(alpha=alpha, l1_ratio=l1_ratio, x=x, y=y)
    
#     optimizer = BayesianOptimization(
#         f=function,
#         pbounds={"alpha": (1e-4, 0.99), "l1_ratio": (1e-4,0.99)},
#         random_state=0,
#         verbose=1,
#     )
#     optimizer.probe(
#     params=[1e-4, 1e-4],
#     lazy=True,
#     )
#     optimizer.maximize(n_iter=25, init_points=20, acq="ei", xi=1e-1)

    
#     #train enet
    
#     return optimizer.max
    # print("Final result:", optimizer.max)        
def append_time_dim(arr, y_, time_stamps):
    """
    apply added time dimension for the data array and label given time_stamps (with downsample_rate=100) in 100ms / need to check with 1375Hz
    """
    time_arr = np.zeros([arr.shape[0]-time_stamps, int(time_stamps*arr.shape[1])])
    for time_idx, time_ in enumerate(np.arange(time_stamps, arr.shape[0])):
        for time_point in range(time_stamps):
            time_arr[time_idx, time_point*arr.shape[1]:(time_point+1)*arr.shape[1]] = arr[time_-time_point,:]
    return time_arr, y_[time_stamps:]

#%%
spoc= SPoC(n_components=1, log=False, reg='oas', transform_into ='average_power', rank='full')
laterality=[("CON"), ("IPS")]
# signal=["ECOG", "STN"]

signal=["STN"]
#clf=LinearRegression(normalize=True, n_jobs=-1)
# clf=LinearRegression()
clf=ElasticNet()

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
    
           
    
           
            
            
            Ypre_tr= OrderedDict()
            score_tr= OrderedDict()
            Ypre_te= OrderedDict()
            score_te= OrderedDict()
            Patterns= OrderedDict()
            Filters= OrderedDict()
            Coef= OrderedDict()
            alpha_param= OrderedDict()
            l1_ratio_param= OrderedDict()
            Label_tr= OrderedDict()
            Label_te= OrderedDict()
    
            for l, mov in enumerate(laterality):
                print("training %s" %mov)
                score_tr[mov] = []
                score_te[mov] = []
                Ypre_tr[mov] = []
                Ypre_te[mov] = []
                Label_tr[mov] = []
                Label_te[mov] = []
                Patterns[mov]=[]
                Filters[mov]=[]
                Coef[mov]=[]
                alpha_param[mov]=[]
                l1_ratio_param[mov]=[]
                if l==0:
                    label=Y_con
                else:
                    label=Y_ips
                
                # #z-scored label also
                # label=stats.zscore(label)
                nfb, nt,nc,ns=np.shape(X)   
                          
                result_lm=[]
                result_rm=[]
                aux=X[0][:,1,:]
                
                label_test=[]
                label_train=[]
                
                onoff_test=[]
                onoff_train=[]
                
                XX=np.swapaxes(X,0,1)
                XX=np.swapaxes(XX,1,2)
                XX=np.swapaxes(XX,2,3)
                XX=XX.astype('float64')
               
                features=FilterBank(estimator=spoc)

                for train_index, test_index in cv.split(aux):
                    Ztr, Zte=label[train_index], label[test_index]
                    #normalize Z
                    # scaler_z = StandardScaler()
                    # scaler_z.fit(Ztr.reshape(-1, 1))
                    # Ztr=np.squeeze(scaler_z.transform(Ztr.reshape(-1, 1)))
                    # Zte=np.squeeze(scaler_z.transform(Zte.reshape(-1, 1)))
                    
                    gtr=features.fit_transform(XX[train_index], Ztr)
                    gte=features.transform(XX[test_index])
                    
                    # for fb in range(nfb): 
                    #     x=X[fb]
                    #     Xtr, Xte=x[train_index,:,:], x[test_index,:,:]
                    #     Xtr=Xtr.astype('float64')
                    #     Xte=Xte.astype('float64')
      
                    #     #fit and transform data
                    #     if fb==0:
                            
                    #         gtr=spoc.fit_transform(Xtr, Ztr)
                            
                    #         filters=spoc.filters_.reshape(-1, spoc.filters_.shape[0],spoc.filters_.shape[1])
                    #         patterns=spoc.patterns_.reshape(-1, spoc.patterns_.shape[0],spoc.patterns_.shape[1])
                            
                    #         gte=spoc.transform(Xte)
                    #     else:                
                    #         gtr=np.hstack((gtr,spoc.fit_transform(Xtr, Ztr)))
                            
                    #         gte=np.hstack((gte,spoc.transform(Xte)))
                            
                    #         ff=spoc.filters_.reshape(-1, filters.shape[1],filters.shape[2])
                    #         pp=spoc.patterns_.reshape(-1, filters.shape[1],filters.shape[2])
                            
                                
                    #         filters=np.vstack((filters,ff))
                    #         patterns=np.vstack((patterns,pp))   
                    
                    # #cropped the values
                    # gtr=np.clip(gtr,-2,2)   
                    # gte=np.clip(gte,-2,2)
                    
                    #standarized
                    # scaler = StandardScaler()
                    # scaler.fit(gtr)
                    # gtr=scaler.transform(gtr)
                    # gte=scaler.transform(gte)
                            
                    dat_tr,label_tr = append_time_dim(gtr, Ztr,time_stamps=5)
                    dat_te,label_te = append_time_dim(gte, Zte,time_stamps=5)
                    
                    scaler = StandardScaler()
                    scaler.fit(dat_tr)
                    dat_tr=scaler.transform(dat_tr)
                    dat_te=scaler.transform(dat_te)
    
                    # Label_te[mov].append(Zte)
                    # Label_tr[mov].append(Ztr)
                    
                    Label_te[mov].append(label_te)
                    Label_tr[mov].append(label_tr)
                    
                    # scaler_z = StandardScaler()
                    # scaler_z.fit(label_tr.reshape(-1, 1))
                    # label_tr=np.squeeze(scaler_z.transform(label_tr.reshape(-1, 1)))
                    # label_te=np.squeeze(scaler_z.transform(label_te.reshape(-1, 1)))
                    
                    optimizer=optimize_enet(x=dat_tr,y=label_tr)
                    # clf=ElasticNet(alpha=optimizer['params']['alpha'], l1_ratio=optimizer['params']['l1_ratio'], max_iter=1000, normalize=False)
                    clf=ElasticNet(alpha=optimizer.x[0], l1_ratio=optimizer.x[1], max_iter=1000)
    
                    
                    clf.fit(dat_tr, label_tr)
                    Ypre_te[mov].append(clf.predict(dat_te))
                    Ypre_tr[mov].append(clf.predict(dat_tr))
                    
                    r2_te=clf.score(dat_te, label_te)
                    if r2_te < 0: r2_te = 0
                    score_te[mov].append(r2_te)
                    r2_tr=clf.score(dat_tr, label_tr)
                    if r2_tr < 0: r2_tr = 0
                    
                    score_tr[mov].append(r2_tr)
                          
                    # Filters[mov].append(filters)
                    # Patterns[mov].append(patterns)
                    
                    Coef[mov].append(clf.coef_)
                    alpha_param[mov].append(clf.alpha)
                    l1_ratio_param[mov].append(clf.l1_ratio)
    
        
                        
            print(np.mean(score_te["CON"]))
           
            #%% save 
            predict_ = {
                "y_pred_test": Ypre_te,
                "y_test": Label_te,
                "y_pred_train": Ypre_tr,
                "y_train": Label_tr,
                "score_tr": score_tr,
                "score_te": score_te,
                "filters": Filters,
                "patterns": Patterns,
                "coef": Coef,
                "classifiers": clf,
                "alpha_param": alpha_param,
                "l1ratio_param": l1_ratio_param,
                "methods": spoc
                
            }
            
            out_path_file = os.path.join(settings['out_path_process']+ settings['num_patients'][s]+'predictions_'+eeg+'_tlag'+'_'+str(subfolder[ss])+'.npy')
            np.save(out_path_file, predict_)        
            
            gc.collect()
            
                
            #%% Plot the True mov and the predicted
            fig, ax = plt.subplots(1, 1, figsize=[10, 4])
            ind_best=np.argmax(score_te['CON'])
            Ypre_te_best=Ypre_te['CON'][ind_best]
            label_test_best=Label_te['CON'][ind_best]
            #times = raw.times[meg_epochs.events[:, 0] - raw.first_samp]
            ax.plot(Ypre_te_best, color='b', label='Predicted mov')
            ax.plot(label_test_best, color='r', label='True mov')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Movement')
            ax.set_title('SPoC mov Predictions')
            ax.text(0.33, 0.9, 'R2={:0.02f}'.format(score_te['CON'][ind_best]),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,fontsize=12) 
            fig.suptitle(eeg+'-Subject_'+ settings['num_patients'][s], fontsize=14, fontweight='bold')
            plt.legend()
            plt.show()
            
            
