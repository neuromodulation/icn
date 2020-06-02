import os
import numpy as np
import settings
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
import multiprocessing
import json
import sys
import settings


VICTORIA = True


settings = {}

if VICTORIA is True:
    # insert at 1, 0 is the script path (or '' in REPL)
    sys.path.insert(1, '/home/victoria/icn/icn_m1')
    settings['BIDS_path'] = "/mnt/Datos/BML_CNCRS/Data_BIDS/"
    settings['out_path'] = "/mnt/Datos/BML_CNCRS/Data_processed/"
else:
    settings['BIDS_path'] = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\"
    settings['out_path'] = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\gen_p_files\\"

settings['resamplingrate']=10
settings['max_dist_cortex']=20
settings['max_dist_subcortex']=5
settings['normalization_time']=10
settings['num_patients']=16

settings['frequencyranges']=[[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]]
settings['seglengths']=[1, 2, 2, 3, 3, 3, 10, 10, 10]


settings['BIDS_path']=settings['BIDS_path'].replace("\\", "/")
settings['out_path']=settings['out_path'].replace("\\", "/")

with open('settings/settings.json', 'w') as fp:
    json.dump(settings, fp)

#%%
def get_int_runs(patient_idx):
    """

    :param patient_idx:
    :return: list with all run files for the given patient
    """
    os.listdir(settings['out_path'])
    if patient_idx < 10:
        subject_id = str('00') + str(patient_idx)
    else:
        subject_id = str('0') + str(patient_idx)
    list_subject = [i for i in os.listdir(settings['out_path']) if i.startswith('sub_'+subject_id) and i.endswith('.p')]
    return list_subject


def get_act_int_list(patient_idx):
    """

    :param patient_idx:
    :return: array in shape (runs_for_patient_idx, 94) including the active grid points in every run
    """

    runs_ = get_int_runs(patient_idx)
    act_ = np.zeros([len(runs_), 94])
    for idx in range(len(runs_)):
        file = open(os.path.join(settings['out_path'], runs_[idx]), 'rb')
        out = pickle.load(file)

        act_[idx, :] = out['arr_act_grid_points']

    return act_

def save_all_act_grid_points():
    """
    function that saves all active grid points in a numpy file --> concatenated as a list over all patients
    can be loaded with act_ = np.load('act_.npy')
    :return:
    """
    l_act = []
    
    for patient_idx in range(settings['num_patients']):
        l_act.append(get_act_int_list(patient_idx))
    np.save('act_.npy', np.array(l_act))
    return np.array(l_act)

def check_leave_out_grid_points(act_, load=True):
    """
    :param act_: array of run files of each patient having interpolated grid points
    :return: a list of grid points which are only occuring in none or one patient
    """
    if load is True:
        return np.load('grid_points_none.npy', allow_pickle=True)
    
    grid_point_occurance = np.zeros(94)
    for patient_idx in range(settings['num_patients']):
        grid_point_occurance[np.where(np.sum(act_[patient_idx], axis=0))[0]] += 1
    grid_points_none = np.where((grid_point_occurance == 0) | (grid_point_occurance == 1))[0]
    np.save('grid_points_none.npy', grid_points_none)
    return grid_points_none

def append_time_dim(arr, y_, time_stamps):
    """
    apply added time dimension for the data array and label given time_stamps (with downsample_rate=100) in 100ms / need to check with 1375Hz
    """
    time_arr = np.zeros([arr.shape[0]-time_stamps, int(time_stamps*arr.shape[1])])
    for time_idx, time_ in enumerate(np.arange(time_stamps, arr.shape[0])):
        for time_point in range(time_stamps):
            time_arr[time_idx, time_point*arr.shape[1]:(time_point+1)*arr.shape[1]] = arr[time_-time_point,:]
    return time_arr, y_[time_stamps:]

def get_train_test_dat(patient_test, grid_point, act_, Train=True, Clip=True):
    """
    For a given grid_point, and a given provided test patient, acquire all combined dat and label information
    
    Parameters
    ----------
    patient_test : TYPE
        DESCRIPTION.
    grid_point : TYPE
        DESCRIPTION.
    act_ : TYPE
        DESCRIPTION.
    Train : TYPE, optional
        etermine if data is returned only from patient_test, or from all other. The default is True.

    Returns
    -------
    dat : array shape(n_data points,n_frequency bands)
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.

    """
    start = 0
    for patient_idx in range(settings['num_patients']):
        if Train is True and patient_idx == patient_test:
            continue
        if Train is False and patient_idx != patient_test:
            continue
        # now load from every patient that has data for that grid point
        if grid_point in np.nonzero(np.sum(act_[patient_idx], axis=0))[0]:
            runs = get_int_runs(patient_idx)
            for run_idx, run in enumerate(runs):
                # does this run has the grid point?
                if act_[patient_idx][run_idx, grid_point] != 0:
                    # load file
                    file = open(os.path.join(settings['out_path'], run), 'rb')
                    out = pickle.load(file)

                    # fill dat
                    if start == 0:
                        dat = out['pf_data_median'][:,grid_point,:]
                        if Clip:
                            dat=np.clip(dat, -2,2)
                        if grid_point < 39 or (grid_point > 78 and grid_point < 86):  # contralateral
                            label = np.squeeze(out['label'][out['label_con_true']==True])
                        else:
                            label = np.squeeze(out['label'][out['label_con_true']==False])
                        start = 1
                    else:
                        dat_new=out['pf_data_median'][:,grid_point,:]
                        if Clip:
                            dat_new=np.clip(dat_new, -2,2)
                        dat = np.concatenate((dat, dat_new), axis=0)

                        if grid_point < 39 or (grid_point > 78 and grid_point < 86):  # contralateral
                            label_new=np.squeeze(out['label'][out['label_con_true']==True])
                            label = np.concatenate((label, label_new), axis=0)
                        else:
                            label_new=np.squeeze(out['label'][out['label_con_true']==False])
                            label = np.concatenate((label, label_new), axis=0)
    return dat, label


def train_grid_point(time_stamps, act_, patient_test, grid_point, model, Verbose=False):
    if Verbose:
        print(grid_point)
    dat, label = get_train_test_dat(patient_test, grid_point, act_, Train=True)
    dat,label = append_time_dim(dat, label,time_stamps)

    dat_test, label_test = get_train_test_dat(patient_test, grid_point, act_, Train=False)
    dat_test,label_test = append_time_dim(dat_test, label_test, time_stamps)

    dat_test = np.clip(dat_test, -2,2)
    dat = np.clip(-2,2)

    model.fit(dat, label)

    y_test_pred = model.predict(dat_test)
    y_train_pred = model.predict(dat)

    predict_ = {
        "y_pred_test": y_test_pred,
        "y_test": label_test,
        "y_pred_train": y_train_pred,
        "y_train": label,
        "r2_test": r2_score(label_test, y_test_pred),
        "r2_train": r2_score(label, y_train_pred)
    }

    return predict_

def run_CV(patient_test, model=LinearRegression(), time_stamps=5):
    """
    given model is trained grid point wise for the provided patient
    saves output estimations and labels in a struct with r2 correlation coefficient
    :param patient_test: CV patient to test
    :param model_fun: provided model function
    :return:
    """
    act_ = np.load('act_.npy', allow_pickle=True)  # load array with active grid points for all patients and runs

    #get all active grid_points for that patient
    arr_active_grid_points = np.zeros(94)
    arr_active_grid_points[np.nonzero(np.sum(act_[patient_test], axis=0))[0]] = 1

    patient_CV_out = np.empty(94, dtype=object)

    for grid_point in np.nonzero(arr_active_grid_points)[0]:
        if grid_point in grid_points_none:
            continue
        patient_CV_out[grid_point] = train_grid_point(time_stamps, act_, patient_test, grid_point, model)
        
    if patient_test < 10:
        subject_id = '00' + str(patient_test)
    else:
        subject_id = '0' + str(patient_test)

    out_path_file = os.path.join(settings['out_path'], subject_id+'prediction.npy')
    np.save(out_path_file, patient_CV_out)

patients=16
    
act_ = save_all_act_grid_points()
grid_points_none = check_leave_out_grid_points(act_, False)

if __name__== "__main__":

    pool = multiprocessing.Pool()
    pool.map(run_CV, np.arange(patients))