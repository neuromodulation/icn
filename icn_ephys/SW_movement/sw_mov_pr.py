import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
import scipy
import mne
import os
import pandas as pd
import numpy as np
import mne
import scipy
import pickle
from sklearn import metrics
import multiprocessing

PATH_TROUGHS = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\SharpWaveAnalysis\\"
PATH_PEAKS = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\SharpWaveAnalysis_Peaks\\"
PATH_SAVE = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\SharpWaveAnalysis\\preprocessed\\"
PATH_COMBINED = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\Combined_runs\\"
PATH_SW_SAVE = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\MOVEMENT DATA\\SW_MOV_PR\\"
dict_metrics_TROUGH = {"sharpness":"mV",
                        "trough":"mV",
                        "interval":"ms",
                        "prominence":"mV",
                        "rise_steepness":"mV'",
                        "rise_time":"ms",
                        "decay_steepness":"mV'",
                        "slope_ratio":"mV'",
                        "decay_time":"ms",
                        "width":"ms"}

dict_metrics_PEAK = {"sharpness":"mV",
                "trough":"mV",
                "interval":"ms",
                "prominence":"mV",
                "rise_steepness":"mV'",
                "rise_time":"ms",
                "decay_steepness":"mV'",
                "slope_ratio":"mV'",
                "decay_time":"ms",
                "width":"ms"}

class NoValidTroughException(Exception):
    pass

def get_sw_pr_ch(sub, loc, key, mov_con, mov_ips, df_TROUGHS):

    thr_space = np.arange(0, 200, 5)
    time_window_space = np.arange(0, 50, 1)
    pr_array = np.ones((thr_space.shape[0],time_window_space.shape[0], mov_con[::100].shape[0]), dtype=bool)

    for thr_idx, thr in enumerate(thr_space):
        prominences_thr = np.zeros(mov_con[::100].shape[0], dtype=bool)
        # here a vector is creates of fs = 10Hz which determines presence of passed thr prominence
        t_arr = np.arange(0, mov_con[::100].shape[0], 1)*100
        for t_idx, t in enumerate(t_arr):
            if np.sum(df_TROUGHS[(df_TROUGHS["trough_idx"] < t) &
                           (df_TROUGHS["trough_idx"] > (t-100))]["prominence"] > thr) > 0:
                prominences_thr[t_idx] = True

        t_arr = np.arange(time_window_space[-1], prominences_thr.shape[0], 1)
        for t_w_idx, time_window in enumerate(time_window_space):
            for t in t_arr: # t is hier in 10 Hz
                pr_array[thr_idx, t_w_idx, t] = False if np.sum(prominences_thr[t-time_window:t]) > 0 else True

    F1_conips = np.zeros([thr_space.shape[0], time_window_space.shape[0]])
    AUC_conips = np.zeros([thr_space.shape[0], time_window_space.shape[0]])
    F1_con = np.zeros([thr_space.shape[0], time_window_space.shape[0]])
    AUC_con = np.zeros([thr_space.shape[0], time_window_space.shape[0]])
    F1_ips = np.zeros([thr_space.shape[0], time_window_space.shape[0]])
    AUC_ips = np.zeros([thr_space.shape[0], time_window_space.shape[0]])

    for thr_i,_ in enumerate(thr_space):
        for tw_i,_ in enumerate(time_window_space):
            F1_conips[thr_i, tw_i] = metrics.f1_score(pr_array[thr_i, tw_i,:],
                                (mov_con[::100]+mov_ips[::100])>0)
            F1_con[thr_i, tw_i] = metrics.f1_score(pr_array[thr_i, tw_i,:],
                                mov_con[::100]>0)
            F1_ips[thr_i, tw_i] = metrics.f1_score(pr_array[thr_i, tw_i,:],
                                mov_ips[::100]>0)
            try:
                AUC_conips[thr_i, tw_i] = metrics.roc_auc_score(pr_array[thr_i, tw_i,:],
                                (mov_con[::100]+mov_ips[::100])>0)
            except:
                AUC_conips[thr_i, tw_i] = 0.5
            try:
                AUC_con[thr_i, tw_i] = metrics.roc_auc_score(pr_array[thr_i, tw_i,:],
                                mov_con[::100]>0)
            except:
                AUC_con[thr_i, tw_i] = 0.5
            try:
                AUC_ips[thr_i, tw_i] = metrics.roc_auc_score(pr_array[thr_i, tw_i,:],
                                mov_ips[::100]>0)
            except:
                AUC_ips[thr_i, tw_i] = 0.5

    idx_best_conips = np.unravel_index(F1_conips.argmax(), F1_conips.shape)
    idx_best_con = np.unravel_index(F1_con.argmax(), F1_conips.shape)
    idx_best_ips = np.unravel_index(F1_ips.argmax(), F1_conips.shape)

    dict_save = {
        "F1_conips_max" : F1_conips.max(),
        "F1_con_max" : F1_con.max(),
        "F1_ips_max" : F1_ips.max(),
        "AUC_conips_max" : AUC_conips.max(),
        "AUC_con_max" : AUC_con.max(),
        "AUC_ips_max" : AUC_ips.max(),
        "thr_best_F1_conips" : idx_best_conips[0],
        "tw_best_F1_conips" : idx_best_conips[1],
        "thr_best_F1_con" : idx_best_con[0],
        "tw_best_F1_con" : idx_best_con[1],
        "thr_best_F1_ips" : idx_best_ips[0],
        "tw_best_F1_ips" : idx_best_ips[1],
        "pred_best_conips" : pr_array[idx_best_conips[0], idx_best_conips[1],:],
        "pred_best_con" : pr_array[idx_best_con[0], idx_best_con[1],:],
        "pred_best_ips" : pr_array[idx_best_ips[0], idx_best_ips[1],:],
        "F1_search_conips" : F1_conips,
        "F1_search_con" : F1_con,
        "F1_search_ips" : F1_ips,
        "AUC_search_conips" : AUC_conips,
        "AUC_search_con" : AUC_con,
        "AUC_search_ips" : AUC_ips,
        "mov_con" : mov_con[::100],
        "mov_ips" : mov_ips[::100],
    }
    np.save(os.path.join(PATH_SW_SAVE,"sub_"+str(sub)+"_loc_"+str(loc)+"_ch_"+str(key)+".npy"), dict_save)


files_combined = os.listdir(PATH_COMBINED)
files_troughs = os.listdir(PATH_TROUGHS)  # well this is only for troughs now, adapt for peaks
subjects = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015']

def get_dat_():
    """
    Generator to yield data for outer pool
    """
    for loc in ["ECOG", "STN"]:
        for sub in subjects:
            files = np.sort(np.array([f for f in files_troughs if loc in f and sub in f]))
            for f_idx, f in enumerate(files):
                key = f[f.find("ch_")+3:f.find(".p")] # from syntax: 'sub_000_ch_ECOG_RIGHT_0.p'
                if os.path.exists(os.path.join(PATH_SW_SAVE,"sub_"+str(sub)+"_loc_"+str(loc)+"_ch_"+str(key)+".npy")) is True:
                    continue
                df_TROUGHS = np.load(os.path.join(PATH_TROUGHS, files[f_idx]), allow_pickle=True)
                res = np.load(os.path.join(PATH_COMBINED, [f for f in files_combined if sub in f][0]), allow_pickle=True)



                mov_con = res[key]["mov_con"]
                mov_ips = res[key]["mov_ips"]
                print("loc: "+str(loc))
                print("sub: "+str(sub))
                print("f: "+str(f))

                yield sub, loc, key, mov_con, mov_ips, df_TROUGHS

if __name__ == "__main__":

    # test: run single one through:


    gen_ = get_dat_()
    #dat = next(gen_)
    #print(dat)
    #get_sw_pr_ch(*dat)

    POOL_ = True

    while POOL_ is True:

        l_ = []
        for i in range(59):
            try:

                dat = next(gen_)
            except:
                pool = multiprocessing.Pool(processes=59)
                pool.starmap(get_sw_pr_ch, l_)
                pool.close()
                pool.join()
                POOL_ = False
            if dat is not None:
                l_.append(dat)
            else:
                print("TERMINATE ITERATIONS, last channel reached")

                break
        if POOL_ is not False:
            pool = multiprocessing.Pool(processes=59)
            pool.starmap(get_sw_pr_ch, l_)
            pool.close()
            pool.join()
