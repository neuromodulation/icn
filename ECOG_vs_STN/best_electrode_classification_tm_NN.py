# -*- coding: utf-8 -*-
"""
Created on Wed 23.09.20

@author: Merk
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
import IO
import os
import multiprocessing
from threading import Thread
from queue import Queue
#import tensorflow
#import tensorflow as tf
#import keras
#from keras.layers import BatchNormalization
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#from keras.optimizers import Adam

import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
#from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras import backend as K
from keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

#from tensorflow.python.keras import backend as K
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
#from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint  did NOT work
#from keras.models import load_model
#from tensorflow.keras.layers import Dense, Activation, Permute, Dropout

from scipy import stats
from collections import OrderedDict
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from bayes_opt import BayesianOptimization
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
import gc
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from xgboost import XGBRegressor

VICTORIA = False
WRITE_OUT_CH_IND = False
USED_MODEL = 2 # 0 - Enet, 1 - XGB, 2 - NN
settings = {}
VERBOSE_ALL = 0

if VICTORIA is True:
    # insert at 1, 0 is the script path (or '' in REPL)
    sys.path.insert(1, '/home/victoria/icn/icn_m1')
    settings['BIDS_path'] = "//mnt/Datos/BML_CNCRS/Data_BIDS_new/"
    settings['out_path'] = "/mnt/Datos/BML_CNCRS/Data_processed/Derivatives/Raw_pipeline/"
    if USED_MODEL==0 :
           settings['out_path_process'] = "/mnt/Datos/BML_CNCRS/Spoc/ECoG_STN/LM_Out/"
    if USED_MODEL==1 :
           settings['out_path_process'] = "/mnt/Datos/BML_CNCRS/Spoc/ECoG_STN/XGB_Out/"
else:
    settings['BIDS_path'] = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\"
    settings['out_path'] = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\derivatives\\Int_old_grid\\"
    settings['out_path_process'] = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\MOVEMENT DATA\\ECoG_STN\\NN_Out_NOCV\\"

settings['frequencyranges']=[[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]]
settings['seglengths']=[1, 2, 2, 3, 3, 3, 10, 10, 10]
settings['num_patients']=['000', '001', '004', '005', '006', '007', '008', '009', '010', '013', '014']
settings['BIDS_path']=settings['BIDS_path'].replace("\\", "/")
settings['out_path']=settings['out_path'].replace("\\", "/")



space_NN = [Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate'),
              Integer(low=1, high=3, name='num_dense_layers'),
              Integer(low=1, high=10, prior='uniform', name='num_input_nodes'),
              Integer(low=1, high=10, name='num_dense_nodes'),
              Categorical(categories=['sigmoid', 'tanh'], name='activation')]

def create_model_NN(learning_rate, num_dense_layers, num_input_nodes, num_dense_nodes, activation):
        """
        Create NN tensorflow with different numbers of hidden layers / hidden units
        """

        #start the model making process and create our first layer
        model = tensorflow.keras.Sequential()
        model.add(Dense(num_input_nodes, input_shape=(40,), activation=activation))

        #create a loop making a new dense layer for the amount passed to this model.
        #naming the layers helps avoid tensorflow error deep in the stack trace.
        for i in range(num_dense_layers):
            name = 'layer_dense_{0}'.format(i+1)
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(num_dense_nodes,
                     activation=activation,
                            name=name
                     ))
        #add our classification layer.
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1,activation='linear'))

        #setup our optimizer and compile
        adam = Adam(lr=learning_rate)
        model.compile(optimizer=adam, loss='mean_squared_error',
                     metrics=['mse'])
        return model


def get_int_runs(subject_id, subfolder):
    """

    :param patient_idx:
    :return: list with all run files for the given patient
    """
    os.listdir(settings['out_path'])

    if 'right' in str(subfolder):
        list_subject = [i for i in os.listdir(settings['out_path']) if i.startswith('sub_'+subject_id+'_sess_right') and i.endswith('.p')]
    else:
        list_subject = [i for i in os.listdir(settings['out_path']) if i.startswith('sub_'+subject_id+'_sess_left') and i.endswith('.p')]

    return list_subject


def append_time_dim(arr, y_, time_stamps):
    """
    apply added time dimension for the data array and label given time_stamps (with downsample_rate=100) in 100ms / need to check with 1375Hz
    """
    time_arr = np.zeros([arr.shape[0]-time_stamps, int(time_stamps*arr.shape[1])])
    for time_idx, time_ in enumerate(np.arange(time_stamps, arr.shape[0])):
        for time_point in range(time_stamps):
            time_arr[time_idx, time_point*arr.shape[1]:(time_point+1)*arr.shape[1]] = arr[time_-time_point,:]
    return time_arr, y_[time_stamps:]

cv = KFold(n_splits=3, shuffle=False)
laterality=[("CON"), ("IPS")]
signal=["ECOG", "STN"]

def get_patient_data():

    for sub_idx in np.arange(0, len(settings['num_patients']), 1):
        list_param = [] # list for pool
        for signal_idx, signal_ in enumerate(signal):
            subject_path=settings['BIDS_path'] + 'sub-' + settings['num_patients'][sub_idx]
            subfolder=IO.get_subfolders(subject_path)

            for sess_idx in range(len(subfolder)):
                if os.path.exists(os.path.join(settings['out_path_process'],
                            settings['num_patients'][sub_idx]+'BestChpredictions_'+signal_+'-'+
                                str(subfolder[sess_idx])+'.npy')) is True:
                    continue
                X=[]
                Y_con=[]
                Y_ips=[]
                list_subject=get_int_runs(settings['num_patients'][sub_idx], subfolder[sess_idx])
                list_subject=sorted(list_subject)
                if signal_=="ECOG":
                    if sub_idx==4 and sess_idx==0: #for sake of comparison with spoc
                        list_subject.pop(0)
                    if sub_idx==4 and sess_idx==1:
                        list_subject.pop(2)

                print('RUNNIN SUBJECT_'+ settings['num_patients'][sub_idx]+ '_SESS_'+ str(subfolder[sess_idx]) + '_SIGNAL_' + signal_)
                for run_idx in range(len(list_subject)):
                    with open(settings['out_path']+ '/'+ list_subject[run_idx], 'rb') as handle:
                        run_ = pickle.load(handle)

                    #concatenate features
                    #get cortex data only
                    if signal_=="ECOG":
                        ind_cortex=run_['used_channels']['cortex']
                        rf=run_['rf_data_median']
                        x=rf[:,ind_cortex,:]
                        x=np.clip(x, -2,2) # this should have been implemented in the pipeline
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

                gc.collect() # free unreferenced memory
                X=np.concatenate(X, axis=0)
                Y_con=np.concatenate(Y_con, axis=0)
                Y_ips=np.concatenate(Y_ips, axis=0)

                for laterality_idx, laterality_ in enumerate(laterality):
                    for ch_idx in range(X.shape[1]):
                        if laterality_ == "CON":
                            label_here = Y_con
                        else:
                            label_here = Y_ips
                        out_path_file = os.path.join(settings['out_path_process']+ \
                            settings['num_patients'][sub_idx]+'BestChpredictions_'+\
                            signal_+'-ch-'+str(ch_idx)+'-lat-'+str(laterality_)+'-'+str(subfolder[sess_idx])+'.npy')
                        if os.path.exists(out_path_file) is True:
                            print("file already exists: "+str(out_path_file))
                            continue
                        yield X[:,ch_idx,:], label_here, ch_idx, laterality_, signal_, subfolder, sess_idx, sub_idx
                        #list_param.append((X[:,ch_idx,:], label_here, ch_idx, laterality_, signal_, subfolder, sess_idx, sub_idx))
        #pool = multiprocessing.Pool(len(list_param))
        #pool.starmap(pool_function_la, list_param)

def pool_function_la(X, label, ch_idx, laterality_, signal_, subfolder, sess_idx, sub_idx):

    def optimize_nn(x,y, ch_idx, laterality_):

        @use_named_args(space_NN)
        def objective(**params):
            print(params)
            learning_rate=params["learning_rate"]
            num_dense_layers=params["num_dense_layers"]
            num_input_nodes=params["num_input_nodes"]
            num_dense_nodes=params["num_dense_nodes"]
            activation=params["activation"]

            with tf.device(tf.DeviceSpec(device_type="CPU")):
                X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.9,shuffle=False)
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8,shuffle=False)
                es = EarlyStopping(monitor='val_mse', mode='min', verbose=VERBOSE_ALL, patience=10)
                mc = ModelCheckpoint('best_model_'+str(ch_idx)+str(laterality_)+'.h5', monitor='val_mse', mode='min', verbose=VERBOSE_ALL, save_best_only=True)
                model = create_model_NN(learning_rate, num_dense_layers, num_input_nodes, num_dense_nodes, activation)
                try:
                    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, batch_size=100, verbose=VERBOSE_ALL, callbacks=[mc,es])
                    model = load_model('best_model_'+str(ch_idx)+str(laterality_)+'.h5', compile = False)
                except Exception as e:
                    # error
                    print(e)
                    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, batch_size=100, verbose=VERBOSE_ALL, callbacks=[es])
                try:
                    sc = metrics.r2_score(model.predict(X_test), y_test)
                except:
                    print("error at r2 evaluation")
                    sc = 0
                if sc < 0: sc = 0
            return -sc

        print("boosting NN "+str(ch_idx)+" lat "+str(laterality_)+ " x shape: "+str(x.shape))
        res_gp = gp_minimize(objective, space_NN, n_calls=10, random_state=0)
        return res_gp

    Ypre_te= []
    Ypre_tr= []
    score_tr= []
    Ypre_te= []
    score_te= []
    label_test=[]
    label_train=[]
    coords = []
    coef_ = []
    hyp_=[]

    Xtr, Xte, Ytr, Yte = train_test_split(X, label, train_size=0.9,shuffle=False)
    label_test.append(Yte)
    label_train.append(Ytr)
    dat_tr,label_tr = append_time_dim(Xtr, Ytr, time_stamps=5)
    dat_te,label_te = append_time_dim(Xte, Yte, time_stamps=5)


    try:
        optimizer = optimize_nn(dat_tr, label_tr, ch_idx, laterality_)
    except Exception as e:
        print(e)
        return None

    learning_rate=optimizer['x'][0]
    num_dense_layers=optimizer['x'][1]
    num_input_nodes=optimizer['x'][2]
    num_dense_nodes=optimizer['x'][3]
    activation=optimizer['x'][4]
    model = create_model_NN(learning_rate, num_dense_layers, num_input_nodes, num_dense_nodes, activation)
    es = EarlyStopping(monitor='val_mse', mode='min', verbose=VERBOSE_ALL, patience=10)
    mc = ModelCheckpoint('best_model_'+str(ch_idx)+str(laterality_)+'.h5', monitor='val_mse', mode='min', verbose=VERBOSE_ALL, save_best_only=True)
    X_train, X_val, y_train, y_val = train_test_split(dat_tr, label_tr, train_size=0.8,shuffle=True)
    try:
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, batch_size=100, verbose=VERBOSE_ALL, callbacks=[mc,es])
        model = load_model('best_model_'+str(ch_idx)+str(laterality_)+'.h5', compile = False)
    except Exception as e:
        # error
        print(e)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, batch_size=100, verbose=VERBOSE_ALL, callbacks=[es])
    try:
        r2_tr = metrics.r2_score(model.predict(X_train), y_train)
    except:
        r2_tr = 0
    if r2_tr < 0: r2_tr = 0
    try:
        r2_te = metrics.r2_score(model.predict(dat_te), label_te)
    except:
        r2_te = 0
    if r2_te < 0: r2_te = 0
    print("channel: "+str(ch_idx)+" r2 test: "+str(r2_te))

    Ypre_te = model.predict(dat_te)[:,0]
    Ypre_tr = model.predict(dat_tr)[:,0]
    hyp_ = optimizer['x']

    predict_ = {
        "y_pred_test": Ypre_te,
        "y_test": label_test,
        "y_pred_train": Ypre_tr,
        "y_train": label_train,
        "score_tr": r2_tr,
        "score_te": r2_te,
        "coef" :coef_,
        "model_hyperparams": hyp_
    }


    out_path_file = os.path.join(settings['out_path_process']+ \
        settings['num_patients'][sub_idx]+'BestChpredictions_'+\
        signal_+'-ch-'+str(ch_idx)+'-lat-'+str(laterality_)+'-'+str(subfolder[sess_idx])+'.npy')
    print("saving dict of path: "+str(out_path_file))
    np.save(out_path_file, predict_)

if __name__ == '__main__':
    #for sub_idx in np.arange(0, len(settings['num_patients']), 1):
    #my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    #tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #NUM_PROCESSES = multiprocessing.cpu_count()-1
    NUM_PROCESSES = 50
    patient_dat_generator = get_patient_data()
    gen_counter = 0
    l_dat_gen = []
    while True:
        dat_ = next(patient_dat_generator, None)
        if dat_ is None:
            print("Final subject iteration reached, None received from generator")
            break
        PATH_HERE = os.path.join(settings['out_path_process']+ \
            settings['num_patients'][dat_[7]]+'BestChpredictions_'+\
            dat_[4]+'-ch-'+str(dat_[2])+'-lat-'+str(dat_[3])+'-'+\
            str(dat_[5][dat_[6]])+'.npy')
        if os.path.exists(PATH_HERE) is True:
            print("PATH exists")
            print(PATH_HERE)
            continue
        #X, label, ch_idx, laterality_, signal_, subfolder, sess_idx, sub_idx)
        pool_function_la(dat_[0], dat_[1], dat_[2], dat_[3], dat_[4], dat_[5], dat_[6], dat_[7])
