# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:55:08 2020

@author: Pilin
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

from sklearn.linear_model import ElasticNet

from collections import OrderedDict
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from bayes_opt import BayesianOptimization
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
import gc
from sklearn.preprocessing import StandardScaler
#%%
settings = {}

settings['BIDS_path'] = "//mnt/Datos/BML_CNCRS/Data_BIDS_new/"
settings['out_path'] = "/mnt/Datos/BML_CNCRS/Data_processed/Derivatives/"
settings['out_path_process'] = "/mnt/Datos/BML_CNCRS/Spoc/ECoG_STN/"


settings['frequencyranges']=[[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]]
settings['seglengths']=[1, 2, 2, 3, 3, 3, 10, 10, 10]
# settings['num_patients']=['000', '004', '005', '007', '008', '009', '010', '013', '014']
settings['num_patients']=['000', '001', '004', '005', '006', '007', '008', '009', '010', '013', '014']


settings['BIDS_path']=settings['BIDS_path'].replace("\\", "/")
settings['out_path']=settings['out_path'].replace("\\", "/")


#%%
space_LM = [Real(0, 1, "uniform", name='alpha'),
           Real(0, 1, "uniform", name='l1_ratio')]
         
def optimize_enet(x,y):

    reg=ElasticNet(max_iter=1000)  
    scaler = StandardScaler()
    clf = make_pipeline(scaler, reg)
            

    @use_named_args(space_LM)
    def objective(**params):
        reg.set_params(**params)
        cval = cross_val_score(clf, x, y, scoring='r2', cv=3)
        cval[np.where(cval < 0)[0]] = 0
    
        return -cval.mean()

    res_gp = gp_minimize(objective, space_LM, n_calls=20)
    return res_gp


def get_int_runs(subject_id, subfolder):
    """

    :param patient_idx:
    :return: list with all run files for the given patient
    """
    os.listdir(settings['out_path'])
    # if patient_idx < 10:
    #     subject_id = str('00') + str(patient_idx)
    # else:
    #     subject_id = str('0') + str(patient_idx)
    if 'right' in str(subfolder):
        list_subject = [i for i in os.listdir(settings['out_path']) if i.startswith('sub_'+subject_id+'_sess_right') and i.endswith('.p')]
    else:
        list_subject = [i for i in os.listdir(settings['out_path']) if i.startswith('sub_'+subject_id+'_sess_left') and i.endswith('.p')]
                                                                                        
    return list_subject

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
#     params=[1e-3, 1e-3],
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
cv = KFold(n_splits=3, shuffle=False)  
laterality=["CON", "IPS"]
signal=["ECOG", "STN"]


#%%cross-val within subject   
len(settings['num_patients'])
for m, eeg in enumerate(signal):  
   

    for s in range(len(settings['num_patients'])):
        subject_path=settings['BIDS_path'] + 'sub-' + settings['num_patients'][s]
        subfolder=IO.get_subfolders(subject_path)
       
        for ss in range(len(subfolder)):
            X=[]
            Y_con=[]
            Y_ips=[]
            list_subject=get_int_runs(settings['num_patients'][s], subfolder[ss])
            list_subject=sorted(list_subject)
            if eeg=="ECOG":
                if s==4 and ss==0: #for sake of comparison with spoc
                    list_subject.pop(0)
                if s==4 and ss==1:
                    list_subject.pop(2)
                  
    
            print('RUNNIN SUBJECT_'+ settings['num_patients'][s]+ '_SESS_'+ str(subfolder[ss]) + '_SIGNAL_' + eeg)
    
            for run_idx in range(len(list_subject)):
                with open(settings['out_path']+ '/'+ list_subject[run_idx], 'rb') as handle:
                    run_ = pickle.load(handle)
                #concatenate features
                #get cortex data only
                if eeg=="ECOG":    
                    ind_cortex=run_['used_channels']['cortex']    
                    rf=run_['rf_data_median']
                    x=rf[:,ind_cortex,:]
                    x=np.clip(x, -2,2)
                    
                    y=run_['label_baseline_corrected']
                    con_true=run_['label_con_true']
                    y_con=np.squeeze(y[con_true==True])
                    y_ips=np.squeeze(y[con_true==False])
                    
                    X.append(x)
                    Y_con.append(y_con)
                    Y_ips.append(y_ips)
            
                
                else:
                    ind_subcortex=run_['used_channels']['subcortex']
                    if ind_subcortex is not None:
                      
                        rf=run_['rf_data_median']
                        x=rf[:,ind_subcortex,:]
                        x=np.clip(x, -2,2)
                        
                        y=run_['label_baseline_corrected']
                        con_true=run_['label_con_true']
                        y_con=np.squeeze(y[con_true==True])
                        y_ips=np.squeeze(y[con_true==False])
                        
                        X.append(x)
                        Y_con.append(y_con)
                        Y_ips.append(y_ips)
            
            gc.collect()
            
            X=np.concatenate(X, axis=0)
            Y_con=np.concatenate(Y_con, axis=0)
            Y_ips=np.concatenate(Y_ips, axis=0)  
                
      
            
            Yp_tr= OrderedDict()
            sc_tr= OrderedDict()
            Yp_te= OrderedDict()
            sc_te= OrderedDict()
           
            Yt_tr= OrderedDict()
            Yt_te= OrderedDict()
    
            for l, mov in enumerate(laterality):
                print("training %s" %mov)
                sc_tr[mov] = []
                sc_te[mov] = []
                Yp_tr[mov] = []
                Yp_te[mov] = []
                Yt_tr[mov] = []
                Yt_te[mov] = []
             
                if mov=="CON":
                    label=Y_con
                else:
                    label=Y_ips
              
                         
                #run CV
                       
                Score_tr=np.empty(X.shape[1], dtype=object)    
                Score_te=np.empty(X.shape[1], dtype=object)     
                Label_te=np.empty(X.shape[1], dtype=object)
                Label_tr=np.empty(X.shape[1], dtype=object)    
                Labelpre_te=np.empty(X.shape[1], dtype=object)
                Labelpre_tr=np.empty(X.shape[1], dtype=object)     
                # AUC_te=np.empty(X.shape[1], dtype=object) 
                # AUC_tr=np.empty(X.shape[1], dtype=object)     
        
            
                #for each electrode
                for e in range(X.shape[1]):
                    
                    
                    dat_,label_ = append_time_dim(X[:,e,:], label,time_stamps=5) 
                    # #z-score with no cv
                    # scaler = StandardScaler()
                    # scaler.fit(dat_)
                    # dat_=scaler.transform(dat_)
                    # #undomment this line (and chenge appropraitely) for having a testing data out
                    # dat_tr, dat_te, label_tr, label_te = train_test_split(dat_, label_, test_size=0.1, random_state=0)

                     
                    optimizer=optimize_enet(x=dat_,y=label_)
                    # model=ElasticNet(alpha=optimizer['params']['alpha'], l1_ratio=optimizer['params']['l1_ratio'], max_iter=1000, normalize=False)
                    
                    # scaler = StandardScaler()
                    # scaler.fit(dat_tr)
                    # dat_tr=scaler.transform(dat_tr)
                    # dat_te=scaler.transform(dat_te)
                    # model=ElasticNet(alpha=optimizer.x[0], l1_ratio=optimizer.x[1], max_iter=1000)
            
                                   
                    # model.fit(dat_tr, label_tr)
                    # Ypre_te=model.predict(dat_te)
                    # Ypre_tr=model.predict(dat_tr)
                    # r2_tr=model.score(dat_tr, label_tr)
                    # if r2_tr < 0: r2_tr = 0
                    # r2_te=model.score(dat_te, label_te)
                    # if r2_te < 0: r2_te = 0
                        
                        # onoff=np.zeros(np.shape(Ytr))
                        # onoff[Ytr>0]=1
                        # auc_tr.append(roc_auc_score(onoff,Ytr))
                        
                        # onoff=np.zeros(np.shape(Yte))
                        # onoff[Yte>0]=1
                        # auc_te.append(roc_auc_score(onoff,Yte))   
                   
                    Score_te[e]=-optimizer.fun
                    # Score_tr[e]=r2_tr
                    # Score_te[e]=r2_te
                    # Label_te[e]=label_te
                    # Label_tr[e]=label_tr
                    # Labelpre_te[e]=Ypre_te
                    # Labelpre_tr[e]=Ypre_tr
                    # AUC_tr[e]=auc_tr
                    # AUC_te[e]=auc_te
        
                # sc_tr[mov] = Score_tr
                sc_te[mov] = Score_te
                # Yp_tr[mov] = Labelpre_te
                # Yp_te[mov] = Labelpre_tr
                # Yt_tr[mov] = Label_te
                # Yt_te[mov] = Label_tr
        
            
            predict_ = {
                    # "y_pred_test": Yp_te,
                    # "y_test": Yt_te,
                    # "y_pred_train": Yp_tr,
                    # "y_train": Yt_tr,
                    # "score_tr": sc_tr,
                    "score_te": sc_te,
                   
                }
                
            # out_path_file = os.path.join(settings['out_path_process']+ settings['num_patients'][s]+'BestChpredictions_'+eeg+'_tlag_CVnorm_'+ str(subfolder[ss])+'.npy')
            # np.save(out_path_file, predict_)      
        

        
        
