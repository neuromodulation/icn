import os
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
import multiprocessing
import json
import sys
import pandas as pd
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn import metrics

# load all data from an interpolated folder into the RAM
settings = {}
settings['Preprocess_path'] = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\derivatives\\Int_dist_25_Median_10\\"
settings['write_path'] = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\derivatives\\res_dist_25_Median_10_CNN\\"
int_runs = os.listdir(settings['Preprocess_path'])  # all runs in the preprocessed path
dat_all = {}
for file in os.listdir(settings['Preprocess_path']):
    with open(os.path.join(settings['Preprocess_path'], file), 'rb') as pickle_file:
        dat_all[file] = pickle.load(pickle_file)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def read_grid():
    cortex_left = np.array(pd.read_csv('C:\\Users\\ICN_admin\\Documents\\icn\\icn_m1\\settings\\cortex_left.tsv', sep="\t"))
    cortex_right = np.array(pd.read_csv('C:\\Users\\ICN_admin\\Documents\\icn\\icn_m1\\settings\\cortex_right.tsv', sep="\t"))
    subcortex_left = np.array(pd.read_csv('C:\\Users\\ICN_admin\\Documents\\icn\\icn_m1\\settings\\subcortex_left.tsv', sep="\t"))
    subcortex_right = np.array(pd.read_csv('C:\\Users\\ICN_admin\\Documents\\icn\\icn_m1\\settings\\subcortex_right.tsv', sep="\t"))
    return cortex_left.T, cortex_right.T, subcortex_left.T, subcortex_right.T

# supply the grid paths individually in read_grid()
cortex_left, cortex_right, subcortex_left, subcortex_right = read_grid()
grid_ = [cortex_left, subcortex_left, cortex_right, subcortex_right]
num_grid_points = np.concatenate(grid_, axis=1).shape[1]
NUM_ECOG_LEFT = grid_[0].shape[1] # 39
NUM_ECOG_RIGHT = grid_[2].shape[1] + NUM_ECOG_LEFT # 78
NUM_SUBCORTEX_LEFT = grid_[1].shape[1] + NUM_ECOG_RIGHT #85

def append_time_dim(arr, y_, time_stamps):
    """
    apply added time dimension for the data array and label given time_stamps (with downsample_rate=100) in 100ms / need to check with 1375Hz
    """
    time_arr = np.zeros([arr.shape[0]-time_stamps, int(time_stamps*arr.shape[1])])
    for time_idx, time_ in enumerate(np.arange(time_stamps, arr.shape[0])):
        for time_point in range(time_stamps):
            time_arr[time_idx, time_point*arr.shape[1]:(time_point+1)*arr.shape[1]] = arr[time_-time_point,:]
    return time_arr, y_[time_stamps:]


def get_train_test_runs(subject_test):
    """read all runs for a given subject, return runs from every other patient as train list as well

    Args:
        subject_test (int): interpolation signature of patient (BIDS_ID)

    Returns:
        list_subject_test, list_subject_train: train and test lists given subject_test
    """
    if subject_test < 10:
        subject_id = str('00') + str(subject_test)
    else:
        subject_id = str('0') + str(subject_test)
    list_subject_test = [i for i in os.listdir(settings['Preprocess_path'])
                         if i.startswith('sub_'+subject_id) and i.endswith('.p')]
    list_subject_train = [i for i in os.listdir(settings['Preprocess_path'])
                          if not i.startswith('sub_'+subject_id) and i.endswith('.p')
                          and not i.startswith("sub_016")]
    return list_subject_test, list_subject_train

def get_train_test_dat(grid_point, int_runs, list_subject_test, list_subject_train, time_stamps=5):
    """read train and test set based on int-runs for a given grid point

    Args:
        grid_point (int):
        int_runs (list): all run strings of preprocessed path
        list_subject_test (list): all test runs
        list_subject_train (list): all train runs
        time_stamps (int, optional): time concatenation parameter. Defaults to 5.

    Returns:
        dat_train, label_train, dat_test, label_test: np arrays
    """
    start_TRAIN = 0
    start_TEST = 0
    for run in int_runs:
        if run in list_subject_test:
            if grid_point in np.nonzero(dat_all[run]["arr_act_grid_points"])[0]:
                if start_TEST == 0:
                    start_TEST = 1
                    dat_test = dat_all[run]["pf_data_median"][:,grid_point,:]
                    if grid_point < NUM_ECOG_LEFT or (grid_point > NUM_ECOG_RIGHT and grid_point < NUM_SUBCORTEX_LEFT):  # contralateral
                        label_test = np.squeeze(dat_all[run]['label_baseline_corrected'][dat_all[run]['label_con_true']==True])
                    else:
                        label_test = np.squeeze(dat_all[run]['label_baseline_corrected'][dat_all[run]['label_con_true']==False])
                else:
                    dat_test = np.concatenate((dat_test,
                                           dat_all[run]["pf_data_median"][:,grid_point,:]), axis=0)
                    if grid_point < NUM_ECOG_LEFT or (grid_point > NUM_ECOG_RIGHT and grid_point < NUM_SUBCORTEX_LEFT):  # contralateral
                        label_new=np.squeeze(dat_all[run]['label_baseline_corrected'][dat_all[run]['label_con_true']==True])
                        label_test = np.concatenate((label_test, label_new), axis=0)
                    else:
                        label_new=np.squeeze(dat_all[run]['label_baseline_corrected'][dat_all[run]['label_con_true']==False])
                        label_test = np.concatenate((label_test, label_new), axis=0)

        elif run in list_subject_train:
            if grid_point in np.nonzero(dat_all[run]["arr_act_grid_points"])[0]:
                if start_TRAIN == 0:
                    start_TRAIN = 1
                    dat_train = dat_all[run]["pf_data_median"][:,grid_point,:]
                    if grid_point < NUM_ECOG_LEFT or (grid_point > NUM_ECOG_RIGHT and grid_point < NUM_SUBCORTEX_LEFT):  # contralateral
                        label_train = np.squeeze(dat_all[run]['label_baseline_corrected'][dat_all[run]['label_con_true']==True])
                    else:
                        label_train = np.squeeze(dat_all[run]['label_baseline_corrected'][dat_all[run]['label_con_true']==False])
                else:
                    dat_train = np.concatenate((dat_train,
                                           dat_all[run]["pf_data_median"][:,grid_point,:]), axis=0)
                    if grid_point < NUM_ECOG_LEFT or (grid_point > NUM_ECOG_RIGHT and grid_point < NUM_SUBCORTEX_LEFT):  # contralateral
                        label_new=np.squeeze(dat_all[run]['label_baseline_corrected'][dat_all[run]['label_con_true']==True])
                        label_train = np.concatenate((label_train, label_new), axis=0)
                    else:
                        label_new=np.squeeze(dat_all[run]['label_baseline_corrected'][dat_all[run]['label_con_true']==False])
                        label_train = np.concatenate((label_train, label_new), axis=0)
    dat_train,label_train = append_time_dim(np.clip(dat_train, -2, 2), label_train,time_stamps)
    dat_test,label_test = append_time_dim(np.clip(dat_test, -2, 2), label_test,time_stamps)
    return dat_train, label_train, dat_test, label_test

def write_CV(subject_test):
    """for a given subject perform the CV; read train and test set for every active grid point
    Write out file as npy
    Args:
        subject_test (int)
    """
    patient_CV_out = np.empty(num_grid_points, dtype=object)
    list_subject_test, list_subject_train = get_train_test_runs(subject_test)
    grid_points_test_used = np.unique(np.concatenate(
            np.array([np.nonzero(dat_all[run_]["arr_act_grid_points"])[0]
            for run_ in list_subject_test])))
    grid_points_train_used = np.unique(np.concatenate(
            np.array([np.nonzero(dat_all[run_]["arr_act_grid_points"])[0]
            for run_ in list_subject_train])))

    for grid_point in grid_points_test_used:
        print("grid_point: "+str(grid_point))
        if grid_point not in grid_points_train_used:
            continue
        dat_train, label_train, dat_test, label_test = \
            get_train_test_dat(grid_point, int_runs,
                               list_subject_test, list_subject_train)

        # split data into train and test sets
        seed = 7
        test_size = 0.33
        X_train, X_val, y_train, y_val = train_test_split(dat_train, label_train, test_size=test_size, random_state=seed)
        model = XGBRegressor(max_depth=20, n_estimators=20)


        #model = LinearRegression()
        #model.fit(dat_train, label_train)

        eval_set = [(X_val, y_val)]
        model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="mae", eval_set=eval_set, verbose=False)

        y_test_pred = model.predict(dat_test)
        y_train_pred = model.predict(X_train)

        patient_CV_out[grid_point] = {
            "y_pred_test": y_test_pred,
            "y_test": label_test,
            "y_pred_train": y_train_pred,
            "y_train": label_train,
            "r2_test": r2_score(label_test, y_test_pred),
            "r2_train": r2_score(y_train, y_train_pred)
        }
    if subject_test < 10:
        subject_id = '00' + str(subject_test)
    else:
        subject_id = '0' + str(subject_test)

    out_path_file = os.path.join(settings['write_path'], subject_id+'prediction.npy')
    np.save(out_path_file, patient_CV_out)

def correlation(x, y):
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den


def define_model():
    model = Sequential()
    model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (8,5,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Flatten())
    model.add(Dense(units = 50 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1 , activation = 'linear'))
    model.compile(optimizer = 'adam' , loss = "mse" , metrics = [correlation])
    model.summary()
    return model


def write_CV_CNN(subject_test):
    patient_CV_out = np.empty(num_grid_points, dtype=object)
    list_subject_test, list_subject_train = get_train_test_runs(subject_test)
    grid_points_test_used = np.unique(np.concatenate(
            np.array([np.nonzero(dat_all[run_]["arr_act_grid_points"])[0]
            for run_ in list_subject_test])))
    grid_points_train_used = np.unique(np.concatenate(
            np.array([np.nonzero(dat_all[run_]["arr_act_grid_points"])[0]
            for run_ in list_subject_train])))

    for grid_point in grid_points_test_used:
        print("grid_point: "+str(grid_point))
        if grid_point not in grid_points_train_used:
            continue
        X_train, y_train, X_test, y_test = \
            get_train_test_dat(grid_point, int_runs,
                               list_subject_test, list_subject_train)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=0)
        X_train = X_train.reshape(X_train.shape[0], 8, 5, order="F")
        X_val = X_val.reshape(X_val.shape[0], 8, 5, order="F")
        X_test = X_test.reshape(X_test.shape[0], 8, 5, order="F")
        X_train = X_train.reshape(-1, 8, 5, 1)
        X_val = X_val.reshape(-1, 8, 5, 1)
        X_test = X_test.reshape(-1, 8, 5, 1)
        model = define_model()
        with tf.device(tf.DeviceSpec(device_type="GPU")):
            es = EarlyStopping(monitor='val_correlation', mode='max', verbose=1, patience=5)
            mc = ModelCheckpoint('best_model.h5', monitor='val_correlation', mode='max', verbose=1, save_best_only=True)
            history = model.fit(X_train, np.round(y_train,2),batch_size = 500, epochs=500,
                      validation_data=(X_val, np.round(y_val,2)), callbacks=[es,mc])

        dependencies = {
            'correlation': correlation
        }
        model = load_model('best_model.h5',custom_objects=dependencies)
        y_test_pred = model.predict(X_test)[:,0]#, ntree_limit=model.best_ntree_limit)
        y_train_pred = model.predict(X_train)[:,0]#, ntree_limit=model.best_ntree_limit)

        ### add here also AUCPR and corr
        precision, recall, thresholds = precision_recall_curve(y_test>0, y_test_pred)
        aucpr_test = metrics.auc(recall, precision)
        precision, recall, thresholds = precision_recall_curve(y_train>0, y_train_pred)
        aucpr_train = metrics.auc(recall, precision)

        patient_CV_out[grid_point] = {
            "y_pred_test": y_test_pred,
            "y_test": y_test,
            "y_pred_train": y_train_pred,
            "y_train": y_train,
            "r2_test": r2_score(y_test, y_test_pred),
            "r2_train": r2_score(y_train, y_train_pred),
            "r_test":np.corrcoef(y_test, y_test_pred)[0][1],
            "r_train":np.corrcoef(y_train, y_train_pred)[0][1],
            "aucpr_test":aucpr_test,
            "aucpr_train":aucpr_train
        }
        print("R TEST: ")
        print(np.corrcoef(y_test, y_test_pred)[0][1])
        print("AUCPR TEST: ")
        print(aucpr_test)
    if subject_test < 10:
        subject_id = '00' + str(subject_test)
    else:
        subject_id = '0' + str(subject_test)

    out_path_file = os.path.join(settings['write_path'], subject_id+'prediction.npy')
    np.save(out_path_file, patient_CV_out)



if __name__ == "__main__":

    NUM_PATIENTS = 16
    for sub in np.arange(0,NUM_PATIENTS,1): #  depending on the number of patients, this can be also parallized using multprocessing pool
        write_CV_CNN(sub)
