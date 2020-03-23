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
from sklearn import linear_model, neural_network, ensemble
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


        for ch in df_elec['name']:
            runs_ = [file for file in tsv_files if 'sub-'+subject_id in file and 'ses-'+ses in file and file.endswith('channels.tsv')]
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

            dict_ch[ch] = {
                "data": ch_dat.tolist(),
                "true_movements": mov_dat.tolist(),  
                "mov_ch": [ch_name for ch_name in data['ch_names'] if 'MOV' in ch_name],
                "choords": np.ndarray.astype(np.array(df_elec[df_elec['name'].str.contains(ch)])[:, 1:4],float).tolist()
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
    apply added time dimension for the data array and label given time_stamps (with downsample_rate=100) in 100ms / need to check with 1375Hz
    """
    time_arr = np.zeros([arr.shape[0]-time_stamps, int(time_stamps*arr.shape[1])])
    for time_idx, time_ in enumerate(np.arange(time_stamps, arr.shape[0])):
        for time_point in range(time_stamps):
            time_arr[time_idx, time_point*arr.shape[1]:(time_point+1)*arr.shape[1]] = arr[time_-time_point,:]
    return time_arr, y_[time_stamps:]

def run_CV_est(subject_id, model_=linear_model.LinearRegression(),out_path = settings.out_path_folder_downsampled, LM_=True, path_data=settings.out_path_folder_downsampled, time_stamps=5):
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
                
                X,y = append_time_dim(X.T, y[mov_idx, :],time_stamps)

                res = np.mean(cross_val_score(model, X, y, scoring='r2', cv=5))
                dict_[ch]["res_"+mov] = {"R2":res}
                if LM_ is True:
                    model = linear_model.LinearRegression()
                    clf = model.fit(X.T,y[0, :])
                    dict_[ch]["res_"+mov]["weight_"+mov] = clf.coef_.tolist()
        with open(out_path+'sub_'+subject_id+'_patient_concat.json', 'w') as fp:
            json.dump(dict_, fp)



if __name__ == "__main__":

    do_LM = True
    #write_patient_concat_ch('013')
    
    subject_id = []
    for patient_idx in np.arange(settings.num_patients):
        if patient_idx < 10:
            subject_id.append(str('00') + str(patient_idx))
        else:
            subject_id.append(str('0') + str(patient_idx))
        print(subject_id)

    if do_LM is True:
        out_here_ = []
        for t in np.arange(100, 1100, 100):
            out_here_.append(out_here = '/home/icn/Documents/raw_out/'+'LM_'+str(t)+'ms')
            if os.path.exists() is False:
                os.mkdir(out_here)
        t = np.arange(1,11,1)    
        pool = multiprocessing.Pool(processes=62)
        model = linear_model.LinearRegression()
        pool.starmap(run_CV_est, zip(subject_id, repeat(model), out_here_, repeat(False), repeat(settings.out_path_folder_downsampled), t))
        #pool.map(partial(run_CV_est, model_=model, out_path=out_here_, LM_=False, time_stamps=t), subject_id)
    else:
        # setup pool for writing json dicts with channel concatenated files
        #pool = multiprocessing.Pool()
        #pool.map(write_patient_concat_ch, subject_id)

        #setup pool for model estimation and writing results back to the JSON dicts

        model = neural_network.MLPRegressor(hidden_layer_sizes=(3,5), activation='relu', shuffle=True, early_stopping=True, max_iter=1000)

        pool = multiprocessing.Pool(processes=62)
        out_here = '/home/icn/Documents/raw_out/NN_3_layer/'
        pool.map(partial(run_CV_est, model_=model, out_path=out_here, LM_=False), subject_id)


        pool = multiprocessing.Pool(processes=62)
        out_here = '/home/icn/Documents/raw_out/RF_/'
        model = ensemble.RandomForestRegressor(n_estimators=200, max_depth=200)
        pool.map(partial(run_CV_est, model_=model, out_path=out_here, LM_=False), subject_id)

        


