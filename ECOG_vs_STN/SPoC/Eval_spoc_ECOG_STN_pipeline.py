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
from sklearn.compose import TransformedTargetRegressor

from TimeLagFilterBank import *
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
space_LM = [Real(1e-6, 1, "uniform", name='alpha'),
           Real(1e-6, 1, "uniform", name='l1_ratio')]
#%%
def func(y, time_stamps=5):
    y_=y.copy()
    # y_[:time_stamps]=np.zeros((time_stamps,1))

    return y_[time_stamps:]

def inverse_func(x, time_stamps=5):
    x_=x.copy()
    x_=np.vstack((np.zeros((time_stamps,1)),x))
    # print(x_.shape)

    return x_
    
time_stamps=5
spoc= SPoC(n_components=1, log=True, reg='oas', transform_into ='average_power', rank='full')
features=TimeLagFilterBank(estimator=spoc)
reg=ElasticNet(max_iter=1000)  
scaler = StandardScaler()
clf = make_pipeline(features,scaler, reg)

regr_trans = TransformedTargetRegressor(regressor=clf,
                                        func=func,
                                        inverse_func=inverse_func, check_inverse=False)
def optimize_enet(x,y):

         

    @use_named_args(space_LM)
    def objective(**params):
        reg.set_params(**params)
        cval = cross_val_score(regr_trans, x, y, scoring='r2', cv=3)
        cval[np.where(cval < 0)[0]] = 0
    
        return -cval.mean()

    res_gp = gp_minimize(objective, space_LM, n_calls=20)
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

#%%
laterality=["CON", "IPS"]
signal=["ECOG", "STN"]
#%%
len(settings['num_patients'])
for m, eeg in enumerate(signal):    

    for s in range(len(settings['num_patients'])):
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
                
                                      
                result_lm=[]
                result_rm=[]
                
                label_test=[]
                label_train=[]
                
                onoff_test=[]
                onoff_train=[]
                
                # #I need to do this for the way the filter bank is implemented
                # XX=np.swapaxes(X,0,1)
                # # del X
                # XX=np.swapaxes(XX,1,2)
                # XX=np.swapaxes(XX,2,3)
                # # XX=XX.astype('float64')
                
              
                
                # #I need to add label to data for time append lags
                # nt, nc,ns,nfb=np.shape(XX)   
                # ll=np.repeat(label.T[np.newaxis,...], nc, axis=0).T   
                # new_data=np.empty((nt,nc, ns+1, nfb))
                # for i in range(nfb):
                #     new_data[:,:,:ns,i]=XX[:,:,:,i]
                #     new_data[:,:,-1,i]=ll
                # # new_data=new_data.astype('float32')
                # # del XX
                # # del ll
                # gc.collect()
                
                 #I need to add label to data for time append lags
                nfb, nt,nc,ns=np.shape(X)   
                ll=np.repeat(label.T[np.newaxis,...], nc, axis=0).T
                # ll=np.reshape(ll, (nt,nc,nfb))
                new_data=[]
                for i in range(nfb):
                    new_data.append(np.dstack((X[i,:,:,:], ll)))
                    # new_data[i,:,-1,:]=ll
                new_data=np.stack(new_data, axis=-1)
                # new_data=new_data.astype('float32')
                gc.collect()
    

                optimizer=optimize_enet(x=new_data,y=label)
                score_te[mov]= -optimizer.fun 
        
                        
            print(score_te[mov])
           
            #%% save 
            predict_ = {
                
                "score_te": score_te,
                "filters":features,
                                
            }
            
            out_path_file = os.path.join(settings['out_path_process']+ settings['num_patients'][s]+'predictions_'+eeg+'_tlag_CV'+'_'+str(subfolder[ss])+'.npy')
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
            
            
