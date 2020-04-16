import os
import numpy as np 
from functools import partial
from itertools import repeat
import pandas as pd
from scipy import stats, signal
import mne
from bids import BIDSLayout
import mne_bids
import settings
import json
from coordinates_io import BIDS_coord
from sklearn import linear_model, neural_network, ensemble, neighbors
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
from sklearn.model_selection import cross_val_score

def write_patient_concat_ch(subject_id, BIDS_path=settings.BIDS_path, path_out=settings.out_path_folder_downsampled, preprocessed_path=settings.out_path_folder):

    layout = BIDSLayout(BIDS_path)
    ses_ = []
    if os.path.exists(os.path.join(BIDS_path, 'sub-'+subject_id, 'ses-right')) is True:
        ses_.append('right')
    if os.path.exists(os.path.join(BIDS_path, 'sub-'+subject_id, 'ses-left')) is True:
        ses_.append('left')

    tsv_files = layout.get(extension='tsv', return_type='filename')

    dict_ch = {}
    for ses in ses_:
        elec_path = os.path.join(BIDS_path, 'sub-'+subject_id, 'ses-'+ses,'eeg','sub-'+subject_id+'_electrodes.tsv')
        df_elec = pd.read_csv(elec_path, sep="\t")

        runs_ = [file for file in tsv_files if 'sub-'+subject_id in file and 'ses-'+ses in file and file.endswith('channels.tsv')]
        for ch in df_elec['name']:
            start = 0
            for run in runs_:
                df_run = pd.read_csv(run, sep="\t")
                if ch in list(df_run['name']):
                    ind_data = np.where(df_run['name'] == ch)[0][0]
                    run_number = run[run.find('run-')+4:run.find('_channels')] # is a string
                    json_raw = 'raw_sub_'+subject_id+'_run_'+run_number+'_sess_'+ses+'.json'

                    with open(os.path.join(preprocessed_path, json_raw), 'r') as fp:
                        data = json.load(fp)

                    if start == 0: 
                        start = 1 
                        ch_dat = np.array(data['data'])[:,ind_data,:]
                        mov_dat = np.array(data['true_movements'])  
                    else:
                        ch_dat = np.concatenate((ch_dat, np.array(data['data'])[:,ind_data,:]), axis=1)
                        mov_dat = np.concatenate((mov_dat, np.array(data['true_movements'])), axis=1)
            if start != 0:
                dict_ch[ch] = {
                    "data": ch_dat.tolist(),
                    "true_movements": mov_dat.tolist(),  
                    "mov_ch": [ch_name for ch_name in data['ch_names'] if 'MOV' in ch_name],
                    "choords": np.array(df_elec.loc[np.where(df_elec['name'] == ch)[0][0]][1:4], float).tolist()
                }
    with open(os.path.join(path_out, 'sub_'+subject_id+'_patient_concat.json'), 'w') as fp:
        json.dump(dict_ch, fp)

def write_all_rawcombined():
    """write for all given patients combined files for all coordinates
    Setup pool for parallel processing 
    """
    subject_id = []
    for patient_idx in np.arange(settings.num_patients):
        if patient_idx < 10:
            subject_id.append(str('00') + str(patient_idx))
        else:
            subject_id.append(str('0') + str(patient_idx))
        
    pool = multiprocessing.Pool()
    pool.map(write_patient_concat_ch, subject_id)

def append_time_dim(arr, y_, time_stamps):
    """
    apply added time dimension for the data array and label given time_stamps
    """
    time_arr = np.zeros([arr.shape[0]-time_stamps, int(time_stamps*arr.shape[1])])
    for time_idx, time_ in enumerate(np.arange(time_stamps, arr.shape[0])):
        for time_point in range(time_stamps):
            time_arr[time_idx, time_point*arr.shape[1]:(time_point+1)*arr.shape[1]] = arr[time_-time_point,:]
    return time_arr, y_[time_stamps:]

def get_movement_idx(ch, mov_channels, Con=True):
    """returns index of mov_channels given boolean Con and ch
    
    Arguments:
        ch {string} -- given channel string
        mov_channels {list} -- string list of used movement channels including LEFT or RIGHT
    
    Keyword Arguments:
        Con {bool} -- laterality (default: {True})
    
    Returns:
        int -- index of mov_channel of the lateral channel
    """
    mov_idx = 0
    if len(mov_channels) > 1:    
        if Con is True:
            if ("RIGHT" in ch and "LEFT" in mov_channels[0]) or \
                ("LEFT" in ch and "RIGHT" in mov_channels[0]):
                mov_idx = 0
            if ("RIGHT" in ch and "LEFT" in mov_channels[1]) or \
                ("LEFT" in ch and "RIGHT" in mov_channels[1]):
                mov_idx = 1
        else:
            if ("RIGHT" in ch and "RIGHT" in mov_channels[0]) or \
                ("LEFT" in ch and "LEFT" in mov_channels[0]):
                mov_idx = 0
            if ("RIGHT" in ch and "RIGHT" in mov_channels[1]) or \
                ("LEFT" in ch and "LEFT" in mov_channels[1]):
                mov_idx = 1
    return mov_idx

def run_CV_est(subject_id, model_=linear_model.LinearRegression(),out_path = settings.out_path_folder_downsampled, LM_=True, classification=False, path_data=settings.out_path_folder_downsampled, time_stamps=5):
    """run a CV baseed on the provided regressor and write out the results in the same dict
    
    Arguments:
        subject_id {[type]} -- [description]
    
    Keyword Arguments:
        model_ {[type]} -- [description] (default: {linear_model.LinearRegression()})
        LM_ {bool} -- [if True, write out LM weights] (default: {True})
    """
    with open(path_data+'sub_'+subject_id+'_patient_concat.json', 'r') as fp:
        dict_ = json.load(fp)
        ch_ = list(dict_.keys())
        for ch in ch_:
            X = np.array(dict_[ch]['data'])
            y = np.array(dict_[ch]['true_movements'])
            for mov_idx, mov in enumerate(dict_[ch]['mov_ch']):
                model = model_
                
                X_,y_ = append_time_dim(X.T, y[mov_idx, :],time_stamps)
                res = np.mean(cross_val_score(model, X_, y_, scoring='r2', cv=5, n_jobs=-1))

                if classification is True:
                    try:
                        res_auc = np.mean(cross_val_score(model, X_, y_>0, scoring='roc_auc', cv=5,n_jobs=-1))
                    except:
                        res_auc = 0.5
                    dict_[ch]["res_"+mov] = {
                        "R2":res, 
                        "AUC": res_auc
                    }
                else:
                    dict_[ch]["res_"+mov] = {
                        "R2":res
                    }

                if LM_ is True:
                    model = linear_model.LinearRegression()
                    clf = model.fit(X_,y_)
                    dict_[ch]["res_"+mov]["weight_"+mov] = clf.coef_.tolist()
        with open(out_path+'sub_'+subject_id+'_patient_concat.json', 'w') as fp:
            json.dump(dict_, fp)

def multi_run_wrapper(args):
    return run_CV_est(*args)


if __name__ == "__main__":

    #write_patient_concat_ch('008')
    #write_all_rawcombined()
    eval_differemt_models =False
    
    #out_here = '/home/icn/Documents/raw_out/'+'LM_100ms/'
    #time_idx = 1
    #run_CV_est('000', model_=linear_model.LinearRegression(), out_path=out_here, LM_=True, path_data=settings.out_path_folder_downsampled, time_stamps=time_idx)
    
    subject_id = []
    for patient_idx in np.arange(settings.num_patients):
        if patient_idx < 10:
            subject_id.append(str('00') + str(patient_idx))
        else:
            subject_id.append(str('0') + str(patient_idx))
        print(subject_id)

    out_here = '/home/icn/Documents/raw_out/RF_32_4_with_AUC/'
    #model = ensemble.RandomForestRegressor(n_estimators=32, max_depth=4)
    #pool = multiprocessing.Pool(processes=62)
    #pool.map(partial(run_CV_est, model_=model, out_path=out_here, LM_=False, path_data=settings.out_path_folder_downsampled, time_stamps=5, classification=True), subject_id)


    for subject_id_ in subject_id:
        model = ensemble.RandomForestRegressor(n_estimators=32, max_depth=4)
        run_CV_est(subject_id_, model_=model, out_path=out_here, LM_=False, path_data=settings.out_path_folder_downsampled, time_stamps=5, classification=True)
    if eval_differemt_models is True:

        use_LM = False
        if use_LM is True:
            time_idx = 0
            for t in np.arange(100, 1100, 100):
                out_here = '/home/icn/Documents/raw_out/'+'LM_'+str(t)+'ms/'
                if os.path.exists(out_here) is False:
                    os.mkdir(out_here)
                pool = multiprocessing.Pool(processes=62)
                model = linear_model.LinearRegression()
                time_idx = time_idx + 1
                pool.map(partial(run_CV_est, model_=model, out_path=out_here, LM_=True, path_data=settings.out_path_folder_downsampled, time_stamps=time_idx), subject_id)
            
            
            pool = multiprocessing.Pool(processes=62)
            model = neighbors.KNeighborsRegressor()
            out_here = '/home/icn/Documents/raw_out/KNN_5_neighbors/'
            if os.path.exists(out_here) is False:
                os.mkdir(out_here)
            pool.map(partial(run_CV_est, model_=model, out_path=out_here, LM_=False, time_stamps=5), subject_id)

            pool = multiprocessing.Pool(processes=62)
            model = neural_network.MLPRegressor(hidden_layer_sizes=(4,), activation='relu', shuffle=True, early_stopping=True, max_iter=1000)
            out_here = '/home/icn/Documents/raw_out/NN_1_4/'
            if os.path.exists(out_here) is False:
                os.mkdir(out_here)
            pool.map(partial(run_CV_est, model_=model, out_path=out_here, LM_=False, time_stamps=5), subject_id)


            pool = multiprocessing.Pool(processes=62)
            model = ensemble.RandomForestRegressor(n_estimators=32, max_depth=4)
            out_here = '/home/icn/Documents/raw_out/RF_32_4/'
            if os.path.exists(out_here) is False:
                os.mkdir(out_here)
            pool.map(partial(run_CV_est, model_=model, out_path=out_here, LM_=False, time_stamps=5), subject_id)


            pool = multiprocessing.Pool(processes=62)
            model = neural_network.MLPRegressor(hidden_layer_sizes=(5,2), activation='relu', shuffle=True, early_stopping=True, max_iter=1000)
            out_here = '/home/icn/Documents/raw_out/NN_2_5/'
            if os.path.exists(out_here) is False:
                os.mkdir(out_here)
            subject_id_here = ['001', '003', '004', '005', '006', '010', '012', '015']
            pool.map(partial(run_CV_est, model_=model, out_path=out_here, LM_=False, time_stamps=5), subject_id_here)


            pool = multiprocessing.Pool(processes=62)
            model = ensemble.RandomForestRegressor(n_estimators=100, max_depth=10)
            out_here = '/home/icn/Documents/raw_out/RF_100_10/'
            if os.path.exists(out_here) is False:
                os.mkdir(out_here)
            pool.map(partial(run_CV_est, model_=model, out_path=out_here, LM_=False, time_stamps=5), subject_id)

