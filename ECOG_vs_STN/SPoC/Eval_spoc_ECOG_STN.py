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
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
import gc
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor

from FilterBank import *
plt.close('all')

#%%
USED_MODEL = 0 # 0 - Enet, 1 - XGB, 2 - NN

settings = {}

settings['BIDS_path'] = "/mnt/Datos/BML_CNCRS/Data_BIDS/"
settings['out_path'] = "/mnt/Datos/BML_CNCRS/Spoc/"
if USED_MODEL==0: settings['out_path_process'] = "/mnt/Datos/BML_CNCRS/Spoc/ECoG_STN/LM_Out_SPoC/"
if USED_MODEL==1: settings['out_path_process'] = "/mnt/Datos/BML_CNCRS/Spoc/ECoG_STN/XGB_Out_SPoC/"



settings['frequencyranges']=[[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]]
settings['seglengths']=[1, 2, 2, 3, 3, 3, 10, 10, 10]
settings['num_patients']=['000', '001', '004', '005', '006', '007', '008', '009', '010', '013', '014']


settings['BIDS_path']=settings['BIDS_path'].replace("\\", "/")
settings['out_path']=settings['out_path'].replace("\\", "/")

#%%
space_LM = [Real(0, 1, "uniform", name='alpha'),Real(0, 1, "uniform", name='l1_ratio')]
space_XGB  = [Integer(1, 100, name='max_depth'),
          Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
          Real(10**0, 10**1, "uniform", name="gamma")]

#%%
# def optimize_enet(x,y):
#     scaler = StandardScaler()
#     reg=ElasticNet(max_iter=1000)
#     clf = make_pipeline(scaler, reg)

      
#     @use_named_args(space_LM)
#     def objective(**params):
#         reg.set_params(**params)
#         cval = cross_val_score(clf, x, y, scoring='r2', cv=3)
#         cval[np.where(cval < 0)[0]] = 0
    
#         return -cval.mean()

#     res_gp = gp_minimize(objective, space_LM, n_calls=20, random_state=0)
#     return res_gp


# def optimize_xgb(x,y):

#     def evalerror(preds, dtrain):
#         """
#         Custom defined r^2 evaluation function
#         """
#         labels = dtrain.get_label()
#         # return a pair metric_name, result. The metric name must not contain a
#         # colon (:) or a space since preds are margin(before logistic
#         # transformation, cutoff at 0)

#         r2 = metrics.r2_score(labels, preds)

#         if r2 < 0:
#             r2 = 0

#         return 'r2', r2

#     @use_named_args(space_XGB)
#     def objective(**params):
#         print(params)

#         params_ = {'max_depth': int(params["max_depth"]),
#              'gamma': params['gamma'],
#              #'n_estimators': int(params["n_estimators"]),
#              'learning_rate': params["learning_rate"],
#              'subsample': 0.8,
#              'eta': 0.1,
#              'disable_default_eval_metric' : 1
#              }
#              #'nthread':59}
#              #'tree_method' : 'gpu_hist'}
#              #'gpu_id' : 1}

#         cv_result = xgb.cv(params_, xgb.DMatrix(x, label=y), num_boost_round=30, feval=evalerror, nfold=3)
#         return -cv_result['test-r2-mean'].iloc[-1]

#     res_gp = gp_minimize(objective, space_XGB, n_calls=20, random_state=0)
#     return res_gp

def enet_train(alpha,l1_ratio,x,y):
    clf=ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000,normalize=False)
    #clf.fit(x,y)
    
    cval = cross_val_score(clf, x, y, scoring='r2', cv=3)
    cval[np.where(cval < 0)[0]] = 0
    return cval.mean()
    
    # return clf.score(x, y)
def optimize_enet(x,y):
    """Apply Bayesian Optimization to select enet parameters."""
    def function(alpha, l1_ratio):
          
        return enet_train(alpha=alpha, l1_ratio=l1_ratio, x=x, y=y)
    
    optimizer = BayesianOptimization(
        f=function,
        pbounds={"alpha": (1e-6, 0.99), "l1_ratio": (1e-6,0.99)},
        random_state=0,
        verbose=1,
    )
    optimizer.probe(
    params=[1e-3, 1e-3],
    lazy=True,
    )
    optimizer.maximize(n_iter=25, init_points=20, acq="ei", xi=1e-1)

    
    #train enet
    
    return optimizer.max
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
spoc= SPoC(n_components=1, log=True, reg='oas', transform_into ='average_power', rank='full')
laterality=["CON", "IPS"]
signal=["ECOG"]

cv = KFold(n_splits=3, shuffle=False)
#%% CV split
len(settings['num_patients'])
for m, eeg in enumerate(signal):    

    for s in range(1,len(settings['num_patients'])):
        gc.collect()
    
        subject_path=settings['BIDS_path'] + 'sub-' + settings['num_patients'][s]
        subfolder=IO.get_subfolders(subject_path)
       
        for ss in range(len(subfolder)):
            X=[] #to append data
            Y_con=[]
            Y_ips=[]
                  
    
            print('RUNNIN SUBJECT_'+ settings['num_patients'][s]+ '_SESS_'+ str(subfolder[ss]) + '_SIGNAL_' + eeg)
    
            list_of_files = os.listdir(settings['out_path']) #list of files in the current directory
            
            file_name=eeg+'_epochs_sub_' + settings['num_patients'][s] + '_sess_'+subfolder[ss][4:]
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
    
            # for results storage           
            Ypre_tr= OrderedDict()
            score_tr= OrderedDict()
            Ypre_te= OrderedDict()
            score_te= OrderedDict()
            Patterns= OrderedDict()
            Filters= OrderedDict()
            Coef= OrderedDict()
            hyperparams= OrderedDict()
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
                hyperparams[mov]=[]
                if l==0:
                    label=Y_con
                else:
                    label=Y_ips
                
                
                result_lm=[]
                result_rm=[]
                
                label_test=[]
                label_train=[]
                
                onoff_test=[]
                onoff_train=[]
                
                                
                nt, nc,ns, nfb=np.shape(X)   
                # adap data for the filter bank implementation 
                # new_data=[]
                # for i in range(nfb):
                #     new_data.append(X[:,:,i,:])
                # new_data=np.stack(new_data, axis=-1).astype('float64')
                gc.collect()
    
               
                features=FilterBank(estimator=spoc)

                for train_index, test_index in cv.split(label):
                    Ztr, Zte=label[train_index], label[test_index]
                    
                    
                    gtr=features.fit_transform(X[train_index], Ztr)
                    gte=features.transform(X[test_index])
                    
                    #cropped the values
                    gtr=np.clip(gtr,-2,2)   
                    gte=np.clip(gte,-2,2)
                            
                    dat_tr,label_tr = append_time_dim(gtr, Ztr,time_stamps=5)
                    dat_te,label_te = append_time_dim(gte, Zte,time_stamps=5)
                    
                    
    
                    # Label_te[mov].append(Zte)
                    # Label_tr[mov].append(Ztr)
                    
                    Label_te[mov].append(label_te)
                    Label_tr[mov].append(label_tr)
                    
                                        
                    if USED_MODEL == 0: # Enet
                            optimizer=optimize_enet(x=dat_tr,y=label_tr)
                            # clf=ElasticNet(alpha=optimizer['x'][0],
                            #                    l1_ratio=optimizer['x'][1],
                            #                    max_iter=1000,
                            #                    normalize=False)
                            clf=ElasticNet(alpha=optimizer['params']['alpha'], l1_ratio=optimizer['params']['l1_ratio'], max_iter=1000)

                            scaler = StandardScaler()
                            scaler.fit(dat_tr)
                            dat_tr=scaler.transform(dat_tr)
                            dat_te=scaler.transform(dat_te)
                    elif USED_MODEL == 1: # XGB
                            optimizer=optimize_xgb(x=dat_tr, y=label_tr)
                            clf=XGBRegressor(max_depth=optimizer['x'][0],
                                               learning_rate=optimizer['x'][1],
                                               gamma=optimizer['x'][2])
                            
                    
                    #now that the LM is fit, scaler training and testing data
                    
                    
                    clf.fit(dat_tr, label_tr)
                    Ypre_te[mov].append(clf.predict(dat_te))
                    Ypre_tr[mov].append(clf.predict(dat_tr))
                    
                    r2_te=clf.score(dat_te, label_te)
                    if r2_te < 0: r2_te = 0
                    score_te[mov].append(r2_te)
                    r2_tr=clf.score(dat_tr, label_tr)
                    if r2_tr < 0: r2_tr = 0
                    
                    score_tr[mov].append(r2_tr)
                          
                    Filters[mov].append(features.filters)
                    Patterns[mov].append(features.patterns)
                    
                    if USED_MODEL == 0: Coef[mov].append(clf.coef_)
                    # hyperparams[mov].append(optimizer['x'])
                    hyperparams[mov].append(optimizer['params'])
    
        
                        
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
                "model_hyperparams": hyperparams,
                "methods": spoc
                
            }
            
            out_path_file = os.path.join(settings['out_path_process']+ settings['num_patients'][s]+'predictions_'+eeg+'_tlag_bopt_'+str(subfolder[ss])+'.npy')
            np.save(out_path_file, predict_)        
            
            gc.collect()
            
                
            # #%% Plot the True mov and the predicted
            # fig, ax = plt.subplots(1, 1, figsize=[10, 4])
            # ind_best=np.argmax(score_te['CON'])
            # Ypre_te_best=Ypre_te['CON'][ind_best]
            # label_test_best=Label_te['CON'][ind_best]
            # #times = raw.times[meg_epochs.events[:, 0] - raw.first_samp]
            # ax.plot(Ypre_te_best, color='b', label='Predicted mov')
            # ax.plot(label_test_best, color='r', label='True mov')
            # ax.set_xlabel('Time (s)')
            # ax.set_ylabel('Movement')
            # ax.set_title('SPoC mov Predictions')
            # ax.text(0.33, 0.9, 'R2={:0.02f}'.format(score_te['CON'][ind_best]),
            # verticalalignment='bottom', horizontalalignment='right',
            # transform=ax.transAxes,fontsize=12) 
            # fig.suptitle(eeg+'-Subject_'+ settings['num_patients'][s], fontsize=14, fontweight='bold')
            # plt.legend()
            # plt.show()
            
            
