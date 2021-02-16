import features 
import numpy as np 
import json
import mne_bids
import run_analysis
import generator
import pandas as pd
import os

if __name__ == "__main__":

    PATH_VHDR = r'C:\Users\ICN_admin\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\Datasets\BIDS_Berlin\rawdata_Berlin\sub-002\ses-20200131\ieeg\sub-002_ses-20200131_task-SelfpacedRotationR_acq-MedOn+StimOff_run-4_ieeg.vhdr'
    PATH_M1 = r'C:\Users\ICN_admin\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\Datasets\BIDS_Berlin\derivatives\sub-002\ses-20200131\ieeg\sub-002_ses-20200131_task-SelfpacedRotationR_acq-MedOn+StimOff_run-4_channels_M1.tsv'

    df_M1 = pd.read_csv(PATH_M1, sep="\t")

    # read settings 
    with open('settings.json', encoding='utf-8') as json_file:
        settings = json.load(json_file)

    entities = mne_bids.get_entities_from_fname(PATH_VHDR)
    bids_path = mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], task=entities["task"], \
        run=entities["run"], acquisition=entities["acquisition"], datatype="ieeg", root=settings["BIDS_path"])#root=settings["BIDS_path"])#
    raw_arr = mne_bids.read_raw_bids(bids_path)
    ieeg_raw = raw_arr.get_data()
    fs = int(np.ceil(raw_arr.info["sfreq"]))
    line_noise = int(raw_arr.info["line_freq"])
    ch_names = df_M1[(df_M1["used"] == 1) & (df_M1["target"] == 0)]["name"]

    LIMIT_ = 30000
    gen = generator.ieeg_raw_generator(ieeg_raw[:,:LIMIT_], df_M1, settings, fs) # clip for timing reasons 

    features_ = features.Features(s=settings, fs=fs, line_noise=line_noise, channels=ch_names)

    # call now run_analysis.py 
    df_ = run_analysis.run(gen, features_, settings, df_M1)

    #resample_label 
    ind_label = np.where(df_M1["target"] == 1)[0]
    dat_ = ieeg_raw[ind_label, int(fs*settings["bandpass_filter_settings"]["segment_lengths"][0]):LIMIT_]
    label_downsampled = dat_[:, ::int(np.ceil(fs / settings["resampling_rate"]))]

    # and add to df 
    if df_.shape[0] == label_downsampled.shape[1]:
        for idx, label_ch in enumerate(df_M1["name"][ind_label]):
            df_[label_ch] = label_downsampled[idx, :]
    else: 
        print("label dimensions don't match, saving downsampled label extra")

    df_.to_pickle(os.path.join(settings["out_path"]))
