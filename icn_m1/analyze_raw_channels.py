import os
import numpy as np 
import pandas as pd
from scipy import stats, signal
#import mne
#from bids import BIDSLayout
#import mne_bids
import settings
import json
#from coordinates_io import BIDS_coord
from sklearn import linear_model
import multiprocessing
from sklearn.model_selection import cross_val_score

def write_patient_concat_ch(subject_id, BIDS_path=settings.BIDS_path, path_out='/Users/hi/Documents/lab_work/data_preprocessed/ch_concat', preprocessed_path='/Users/hi/Documents/lab_work/data_preprocessed/'):

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
    

if __name__ == "__main__":
    
    subject_id = []
    for patient_idx in np.arange(17):
        if patient_idx < 10:
            subject_id.append(str('00') + str(patient_idx))
        else:
            subject_id.append(str('0') + str(patient_idx))
        
    
    