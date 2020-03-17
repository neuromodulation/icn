import numpy as np
import thesis_get_dat
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

patients_in_point = np.load('../final/helper_files/patients_in_point.npy', allow_pickle=True)
patient_arr = np.load('../final/helper_files/patient_arr.npy')
act_arr = np.load('../final/helper_files/act_arr.npy')
units_to_use = np.ones([8, 94])
for unit in range(38, 78):
    units_to_use[5:8, unit] = 0


def get_x_y_datasets(unit, patients_in_unit, patient_test, units_to_use, Train=True):
    '''
    given a grid point return the concatenated array of all patients in that unit without the test patient
    When Train = False: return the concatenated datastrem for the test patient
    units to use: just gamma ECoG grid points are left out
    '''
    start_ = 0
    for patient_train_idx, patient_train in enumerate(patients_in_unit):
        if Train == True:
            if patient_train == patient_test:
                continue
        else:
            if patient_train != patient_test:
                continue
        for sess in np.where(patient_arr == patient_train)[0]:
            if unit in np.nonzero(act_arr[sess,:])[0]:
                mov = np.load('../final/corr_stream/mov_session_'+str(sess)+'.npy')
                dat = np.load('../final/corr_stream/stream_session_'+str(sess)+'_unit_'+str(unit)+'.npy')[:,np.nonzero(units_to_use[:,unit])[0]]
                if start_ == 0:
                    start_ = 1; mov_tot = mov; dat_tot = dat
                else:
                    mov_tot = np.concatenate((mov_tot, mov), axis=0)
                    dat_tot = np.concatenate((dat_tot, dat), axis=0)
    x_tr = dat_tot; y_tr = mov_tot
    return x_tr, y_tr

def get_epochs(y_tr, y_tr_pred, epoch_lim=20, threshold = 0.3):
    '''
    calculate epochs given a prediction and true datastream, the threshold is applied to the true label stream around the -epoch to +epoch_lim
    '''
    if len(y_tr) == 0:
        return False
    ind_mov = np.where(np.diff(np.array(y_tr>threshold)*1) == 1)[0]
    try:
        if not ind_mov:
            return False
    except:
        pass
    low_limit = ind_mov>epoch_lim
    up_limit = ind_mov < y_tr.shape[0]-epoch_lim
    ind_mov = ind_mov[low_limit & up_limit]
    if ind_mov.shape[0] == 0:
        return False
    y_arr = np.zeros([ind_mov.shape[0],int(epoch_lim*2)])
    y_arr_pred = np.zeros([ind_mov.shape[0],int(epoch_lim*2)])
    for idx, i in enumerate(ind_mov):
        y_arr[idx,:] = y_tr[i-epoch_lim:i+epoch_lim]
        y_arr_pred[idx,:] = y_tr_pred[i-epoch_lim:i+epoch_lim]
    return y_arr.T, y_arr_pred.T

def plot_epochs(y_epoch_ipsi_true, y_epoch_ipsi_pred, y_epoch_contra_true, y_epoch_contra_pred, epoch_lim = 20):
    '''
    function that con/ips movement and pred epochs around a certain epoch_lim (time)
    '''
    xlab = np.arange(-epoch_lim, epoch_lim, 10)
    plt.figure(figsize=(10,7))
    plt.subplot(221)
    plt.imshow(y_epoch_ipsi_true.T, aspect='auto')
    plt.xticks(np.arange(0,epoch_lim*2,10), xlab*0.1)
    plt.ylabel('Movements')
    plt.xlabel('Time [s]')
    plt.title('True labels ipsilateral')
    plt.colorbar()
    plt.clim(-1, 1)

    plt.subplot(222)
    plt.imshow(y_epoch_ipsi_pred.T, aspect='auto')
    plt.xticks(np.arange(0,epoch_lim*2,10), xlab*0.1)
    plt.ylabel('Movements')
    plt.xlabel('Time [s]')
    plt.title('Predictions ipsilateral')
    plt.colorbar()
    plt.clim(-1, 1)

    plt.subplot(223)
    plt.imshow(y_epoch_contra_true.T, aspect='auto')
    plt.xticks(np.arange(0,epoch_lim*2,10), xlab*0.1)
    plt.ylabel('Movements')
    plt.xlabel('Time [s]')
    plt.title('True labels contralateral')
    plt.colorbar()
    plt.clim(-1, 1)

    plt.subplot(224)
    plt.imshow(y_epoch_contra_pred.T, aspect='auto')
    plt.xticks(np.arange(0,epoch_lim*2,10), xlab*0.1)
    plt.ylabel('Movements')
    plt.xlabel('Time [s]')
    plt.title('Predictions contralateral')
    plt.colorbar()
    plt.clim(-1, 1)
    plt.tight_layout()

def plot_movement_traces(y_epoch_ipsi_true, y_epoch_ipsi_pred, y_epoch_contra_true, y_epoch_contra_pred, epoch_lim = 20):
    '''
    plot con/ips mean epochs mov/pred
    '''
    xlab = np.arange(-epoch_lim, epoch_lim, 10)
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    epoch_y_te_pr = y_epoch_contra_pred.T
    epoch_y_te = y_epoch_contra_true.T

    plt.title('Contralateral movements')
    plt.errorbar(np.arange(epoch_y_te_pr.shape[1]), epoch_y_te_pr.mean(axis=0), epoch_y_te_pr.std(axis=0), \
                label='predict', alpha=.85)
    plt.errorbar(np.arange(epoch_y_te.shape[1]), epoch_y_te.mean(axis=0), epoch_y_te.std(axis=0), label='true', alpha=0.85)
    plt.xticks(np.arange(0,epoch_lim*2,10), xlab*0.1)
    plt.xlabel('Time [s]')
    plt.ylim(-0.5, 3)
    plt.ylabel('Force')
    plt.legend()

    plt.subplot(122)
    epoch_y_te_pr = y_epoch_ipsi_pred.T
    epoch_y_te = y_epoch_ipsi_true.T

    plt.title('Ipsilateral movements')
    plt.errorbar(np.arange(epoch_y_te_pr.shape[1]), epoch_y_te_pr.mean(axis=0), epoch_y_te_pr.std(axis=0), \
                label='predict', alpha=.85)
    plt.errorbar(np.arange(epoch_y_te.shape[1]), epoch_y_te.mean(axis=0), epoch_y_te.std(axis=0), label='true', alpha=0.85)
    plt.xticks(np.arange(0,epoch_lim*2,10), xlab*0.1)
    plt.xlabel('Time [s]')
    plt.legend()
    plt.ylim(-0.5, 3)
    plt.ylabel('Force')
    plt.tight_layout()

def read_prec_npv_auc_f1_sess(threshold_arr, patient_arr, act_arr, Use_Train_data=True, folder='RF_32_est_max_depth_4_time_5'):
    '''
    calculate for every patient and every threshold Prec/NPV/AUC/F1 the score
    when Use Train =True, training performance is read out for the given folder
    '''
    auc_con_thr = []; auc_con = []
    auc_ips_thr = []; auc_ips = []
    f1_con_thr = []; f1_con = []
    f1_ips_thr = []; f1_ips = []
    prec_con_thr = []; prec_con = []
    prec_ips_thr = []; prec_ips = []
    npv_con_thr = []; npv_con = []
    npv_ips_thr = []; npv_ips = []

    for thr_idx, thr in enumerate(threshold_arr):
        auc_con = []; auc_ips = []; f1_con = []; f1_ips = []
        prec_con = []; prec_ips = []; npv_con = []; npv_ips = []
        for patient_idx in range(16):

            if Use_Train_data == False:
                y_pred_con = y_best_con[patient_idx]
                y_true_con = mov_best_con[patient_idx]

                y_pred_ips = y_best_ips[patient_idx]
                y_true_ips = mov_best_ips[patient_idx]
            else:
                best_grid_point_con = best_con[patient_idx]
                y_pred_con = np.load(str(folder)+'/y_training_predict_'+str(best_grid_point_con)+'_patient_'+str(patient_idx)+'.npy')
                patients_in_unit = patients_in_point[best_grid_point_con]
                _, y_true_con = get_x_y_datasets(best_grid_point_con, patients_in_unit, patient_idx, units_to_use, Train=True)
                y_true_con = y_true_con[time_stamps:]

                best_grid_point_ips = best_ips[patient_idx]
                y_pred_ips = np.load(str(folder)+'/y_training_predict_'+str(best_grid_point_ips)+'_patient_'+str(patient_idx)+'.npy')
                patients_in_unit = patients_in_point[best_grid_point_ips]
                _, y_true_ips = get_x_y_datasets(best_grid_point_ips, patients_in_unit, patient_idx, units_to_use, Train=True)
                y_true_ips = y_true_ips[time_stamps:]


            auc_con.append(metrics.roc_auc_score(y_true_con>0, y_pred_con>thr))
            f1_con.append(metrics.f1_score(y_true_con>0, y_pred_con>thr))
            conf_mat = metrics.confusion_matrix(y_true_con>0, y_pred_con>thr)
            prec_con.append(conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1]))
            npv_con.append(conf_mat[0][0] / (conf_mat[0][1] + conf_mat[0][0]))

            auc_ips.append(roc_auc_score(y_true_ips>0, y_pred_ips>thr))
            f1_ips.append(metrics.f1_score(y_true_ips>0, y_pred_ips>thr))
            conf_mat = metrics.confusion_matrix(y_true_ips>0, y_pred_ips>thr)
            prec_ips.append(conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1]))
            npv_ips.append(conf_mat[0][0] / (conf_mat[0][1] + conf_mat[0][0]))

        auc_con_thr.append(auc_con); auc_ips_thr.append(auc_ips)
        f1_con_thr.append(f1_con); f1_ips_thr.append(f1_ips)
        prec_con_thr.append(prec_con); prec_ips_thr.append(prec_ips)
        npv_con_thr.append(npv_con); npv_ips_thr.append(npv_ips)

    return auc_con_thr, auc_ips_thr, f1_con_thr, f1_ips_thr, prec_con_thr, prec_ips_thr, npv_con_thr, npv_ips_thr

def get_rate_grid_search(y_epoch_ipsi_true, y_epoch_ipsi_pred, y_epoch_contra_true, y_epoch_contra_pred, threshold, time_arr = np.arange(-2, 2, 0.1), time_max =12):
    '''
    This function returns a grid search arr for con/ips given required cons. time. prediction time_max and the threshold arr that are tested
    '''
    mov_true_con = np.zeros([y_epoch_contra_true.shape[1], time_max, threshold.shape[0]])
    mov_true_con_time = np.zeros([y_epoch_contra_true.shape[1], time_max, threshold.shape[0]])
    for thr_idx, thr in enumerate(threshold):
        for n in range(time_max):
            for epoch in range(y_epoch_contra_pred.shape[1]):
                counter = 0
                for time in range(y_epoch_contra_pred[:,epoch].shape[0]):
                    if y_epoch_contra_pred[time,epoch] > thr:
                        counter += 1
                    if counter > n:
                        mov_true_con[epoch, n, thr_idx] = 1
                        mov_true_con_time[epoch, n, thr_idx] = time_arr[time]
                        break

    mov_true_ips = np.zeros([y_epoch_ipsi_true.shape[1], time_max, threshold.shape[0]])
    mov_true_ips_time = np.zeros([y_epoch_ipsi_true.shape[1], time_max, threshold.shape[0]])
    for thr_idx, thr in enumerate(threshold):
        for n in range(time_max): #zeit wie viele konsekutive Punkte ermittelt werden müssen
            for epoch in range(y_epoch_ipsi_pred.shape[1]):
                counter = 0
                for time in range(y_epoch_ipsi_pred[:,epoch].shape[0]):
                    if y_epoch_ipsi_pred[time,epoch] > thr:
                        counter += 1
                    if counter > n:
                        mov_true_ips[epoch, n, thr_idx] = 1
                        mov_true_ips_time[epoch, n, thr_idx] = time_arr[time]
                        break
    rate_ips = np.count_nonzero(mov_true_ips, axis=0) / mov_true_ips.shape[0]
    rate_con = np.count_nonzero(mov_true_con, axis=0) / mov_true_con.shape[0]

    return rate_con, rate_ips, mov_true_con_time, mov_true_ips_time

def plot_detection_rate(rate_con, rate_ips, threshold, time_max=12):
    '''
    plot con. and ips. rate grid search
    '''
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    time_ = np.arange(0, time_max)*0.1
    plt.imshow(rate_ips[:,:12], aspect='auto')
    plt.title('Ipsilateral detection rate in dependency of threshold and prediction time')
    plt.xlabel('Threshold')
    plt.xticks(np.arange(threshold[:12].shape[0]), np.round(threshold,2)[:12], rotation=45)
    plt.ylabel('Required consecutive prediction time')
    plt.yticks(np.arange(time_.shape[0]), np.round(time_,2))
    cbar= plt.colorbar()
    plt.clim(0, 1)
    cbar.set_label('Detection rate')
    plt.subplot(122)
    plt.title('Contralateral detection rate in dependency of threshold and prediction time')
    plt.imshow(rate_con, aspect='auto')
    plt.xlabel('Threshold')
    plt.xticks(np.arange(threshold.shape[0]), np.round(threshold,2), rotation=45)
    plt.ylabel('Required consecutive prediction time')
    plt.yticks(np.arange(time_.shape[0]), np.round(time_,2))
    cbar = plt.colorbar()
    plt.clim(0, 1)
    cbar.set_label('Detection rate')
    plt.tight_layout()

def plot_precision_npv(prec_con_thr, prec_ips_thr, npv_con_thr, npv_ips_thr, thr_ = np.arange(0, 2, 0.05)):
    '''
    plot precision and npv with respect to the given threshod array
    '''
    plt.figure(figsize=(12,10))
    plt.subplot(221)
    plt.boxplot(prec_con_thr)
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision contralateral')
    plt.ylim(0,1)
    plt.xticks(range(1, thr_.shape[0]+1), np.round(thr_, 2), rotation=45)

    plt.subplot(222)
    plt.boxplot(prec_ips_thr)
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision ipsilateral')
    plt.ylim(0,1)
    plt.xticks(range(1, thr_.shape[0]+1), np.round(thr_, 2), rotation=45)

    plt.subplot(223)
    plt.boxplot(npv_con_thr)
    plt.xlabel('Threshold')
    plt.ylabel('Negative predicitve value')
    plt.title('Negative predicitve value contralateral')
    plt.ylim(0,1)
    plt.xticks(range(1, thr_.shape[0]+1), np.round(thr_, 2), rotation=45)

    plt.subplot(224)
    plt.boxplot(npv_ips_thr)
    plt.xlabel('Threshold')
    plt.ylabel('Negative predicitve value')
    plt.title('Negative predicitve value ipsilateral')
    plt.xticks(range(1, thr_.shape[0]+1), np.round(thr_, 2), rotation=45)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    #REMARK: FINAL_NN_test beinhaltet den unit grid search

    #plot given epochs
    plot_epochs(epoch_ips_tr, epoch_ips_pr, epoch_con_tr, epoch_con_pr, epoch_lim = 20)

    #plot mean epochs
    plot_movement_traces(epoch_ips_tr, epoch_ips_pr, epoch_con_tr, epoch_con_pr, epoch_lim = 20)

    #Usage of AUC/F1/Prec/NPV calculation
    auc_con_thr, auc_ips_thr, f1_con_thr, f1_ips_thr, prec_con_thr, prec_ips_thr, npv_con_thr, npv_ips_thr = \
        read_prec_npv_auc_f1_sess(threshold, thesis_get_dat.patient_arr, thesis_get_dat.act_arr)

    #plot Precision and NPV
    plot_precision_npv(prec_con_thr, prec_ips_thr, npv_con_thr, \
          npv_ips_thr, thr_ = threshold)

    #PLOT AUC, NPV and Prec with respect to threshold
    plt.figure(figsize=(12,5))
    plt.plot(threshold, [np.mean(npv_ips_thr[thr]) for thr in range(threshold.shape[0])], label='npv ips')
    plt.plot(threshold, [np.mean(npv_con_thr[thr]) for thr in range(threshold.shape[0])], label='npv con')
    plt.plot(threshold, [np.mean(prec_ips_thr[thr]) for thr in range(threshold.shape[0])], label='prec ips')
    plt.plot(threshold, [np.mean(prec_con_thr[thr]) for thr in range(threshold.shape[0])], label='prec con')
    plt.plot(threshold, [np.mean(auc_ips_thr[thr]) for thr in range(threshold.shape[0])], label='auc ips')
    plt.plot(threshold, [np.mean(auc_con_thr[thr]) for thr in range(threshold.shape[0])], label='auc con')
    plt.legend(loc='lower right')
    plt.xlabel('Threshold')
    plt.ylabel('NPV/Precision/AUC')
    plt.title('Precision, NPV, AUC threshold comparison')
    plt.grid(True)


    #CALCULATE the Rate_Con and rate_ips grid search for the best CV training grid point
    best_con = [50, 54, 50, 64, 47, 47, 92, 92, 40, 56, 51, 55, 67, 39, 48, 55]
    best_ips = [2, 15, 11, 3, 2, 11, 19, 14, 7, 14, 12, 16, 8, 5, 17, 10]

    time_stamps=5
    threshold_mov = 0
    time_max = 12

    rate_con_arr = np.zeros([16, time_max, threshold.shape[0]]); rate_ips_arr = np.zeros([16, time_max, threshold.shape[0]])

    for patient_idx in range(16):
        print(patient_idx)
        best_grid_point_con = best_con[patient_idx]
        y_pr_con = np.load(str(folder)+'/y_training_predict_'+str(best_grid_point_con)+'_patient_'+str(patient_idx)+'.npy')
        patients_in_unit = patients_in_point[best_grid_point_con]
        _, y_true_con = get_x_y_datasets(best_grid_point_con, patients_in_unit, patient_idx, units_to_use, Train=True)
        y_true_con = y_true_con[time_stamps:]

        best_grid_point_ips = best_ips[patient_idx]
        y_pr_ips = np.load(str(folder)+'/y_training_predict_'+str(best_grid_point_ips)+'_patient_'+str(patient_idx)+'.npy')
        patients_in_unit = patients_in_point[best_grid_point_ips]
        _, y_true_ips = get_x_y_datasets(best_grid_point_ips, patients_in_unit, patient_idx, units_to_use, Train=True)
        y_true_ips = y_true_ips[time_stamps:]
        #mit y_true und predict jetzt den grid search machen

        epoch_con_tr, epoch_con_pr = get_epochs(y_true_con, y_pr_con, threshold=threshold_mov, epoch_lim=20)
        epoch_ips_tr, epoch_ips_pr = get_epochs(y_true_ips, y_pr_ips, threshold=threshold_mov, epoch_lim=20)

        #berechnen der Detection Rate
        rate_con, rate_ips, mov_true_con_time, mov_true_ips_time = \
                        get_rate_grid_search(epoch_ips_tr, epoch_ips_pr, \
                         epoch_con_tr, epoch_con_pr, threshold, time_arr, time_max =time_max)
        rate_con_arr[patient_idx, :,:] = rate_con; rate_ips_arr[patient_idx, :,:] = rate_ips;

    #CALCULATE the best threshold out of the Training rate_con_arr and ips array
    thr_best = np.zeros([16, time_max])
    for patient_idx in range(16):
        matr = (rate_con_arr[patient_idx,:,:]+rate_ips_arr[patient_idx,:,:]) + \
            np.tile(np.mean(np.array(npv_ips_thr), axis=1), (time_max, 1)) + \
            np.tile(np.mean(np.array(npv_con_thr), axis=1), (time_max, 1))
        thr_best[patient_idx,:] = threshold[np.argmax(matr, axis=1)]

    #PLot best_thr
    plt.imshow(thr_best, aspect='auto')
    plt.colorbar()

    #ESTIMATE FINAL TEST set results
    #gegeben dieser Threshold Werte, ermittle dann die con/ips performances based on the test prediction
    npv_con = np.zeros([16,time_max])
    npv_ips = np.zeros([16, time_max])
    prec_con = np.zeros([16,time_max])
    prec_ips = np.zeros([16, time_max])
    auc_con = np.zeros([16,time_max])
    auc_ips = np.zeros([16, time_max])
    f1_con = np.zeros([16,time_max])
    f1_ips = np.zeros([16, time_max])
    rate_best_con = np.zeros([16,time_max])
    rate_best_ips = np.zeros([16,time_max])

    for patient_idx in range(16):
        print(patient_idx)
        best_grid_point_con = best_con[patient_idx]
        best_grid_point_ips = best_ips[patient_idx]

        y_pr_con, mov_con = get_concat_grid_point(patient_idx, best_grid_point_con, folder='RF_32_est_max_depth_4', lag=5)
        y_pr_ips, mov_ips = get_concat_grid_point(patient_idx, best_grid_point_ips, folder='RF_32_est_max_depth_4', lag=5)

        epoch_con_tr, epoch_con_pr = get_epochs(mov_con, y_pr_con, threshold=threshold_mov, epoch_lim=20)
        epoch_ips_tr, epoch_ips_pr = get_epochs(mov_ips, y_pr_ips, threshold=threshold_mov, epoch_lim=20)

        #berechnen der Detection Rate
        rate_con, rate_ips, mov_true_con_time, mov_true_ips_time = \
                        analyze_classification.get_rate_grid_search(epoch_ips_tr, epoch_ips_pr, \
                         epoch_con_tr, epoch_con_pr, threshold, time_arr, time_max =time_max)


        for cons_time in range(12):
            thr = thr_best[patient_idx, cons_time]
            conf_mat = metrics.confusion_matrix(mov_con>0, y_pr_con>thr)
            prec_con[patient_idx, cons_time] = conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1])
            npv_con[patient_idx, cons_time] = conf_mat[0][0] / (conf_mat[0][1] + conf_mat[0][0])
            auc_con[patient_idx, cons_time] = metrics.roc_auc_score(mov_con>0, y_pr_con>thr)
            f1_con[patient_idx, cons_time] = metrics.f1_score(mov_con>0, y_pr_con>thr)

            conf_mat = metrics.confusion_matrix(mov_ips>0, y_pr_ips>thr)
            prec_ips[patient_idx, cons_time] = conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1])
            npv_ips[patient_idx, cons_time] = conf_mat[0][0] / (conf_mat[0][1] + conf_mat[0][0])
            auc_ips[patient_idx, cons_time] = metrics.roc_auc_score(mov_ips>0, y_pr_ips>thr)
            f1_ips[patient_idx, cons_time] = metrics.f1_score(mov_ips>0, y_pr_ips>thr)

            thr_idx = np.where(threshold == thr)[0][0]
            rate_best_con[patient_idx, cons_time] = rate_con[cons_time, thr_idx]
            rate_best_ips[patient_idx, cons_time] = rate_ips[cons_time, thr_idx]

    #PLOT some of those results
    plt.figure(figsize=(8,8))
    plt.subplot(221)
    plt.imshow(rate_best_con, aspect='auto')
    plt.title('contralateral detection rate')
    cbar = plt.colorbar()
    plt.xticks(np.arange(0,12, 1), np.round(np.arange(1,13, 1)*0.1,2))
    plt.xlabel('required consectutive prediction time')
    plt.ylabel('patients')

    plt.subplot(222)
    plt.imshow(rate_best_ips, aspect='auto')
    plt.title('ipsilateral detection rate')
    cbar = plt.colorbar()
    plt.xticks(np.arange(0,12, 1), np.round(np.arange(1,13, 1)*0.1,2))
    plt.xlabel('required consectutive prediction time')
    plt.ylabel('patients')

    plt.subplot(223)
    plt.imshow(npv_con, aspect='auto')
    plt.title('contralateral NPV')
    cbar = plt.colorbar()
    plt.xticks(np.arange(0,12, 1), np.round(np.arange(1,13, 1)*0.1,2))
    plt.xlabel('required consectutive prediction time')
    plt.ylabel('patients')

    plt.subplot(224)
    plt.imshow(npv_ips, aspect='auto')
    plt.title('ipsilateral NPV')
    cbar = plt.colorbar()
    plt.xticks(np.arange(0,12, 1), np.round(np.arange(1,13, 1)*0.1,2))
    plt.xlabel('required consectutive prediction time')
    plt.ylabel('patients')

    plt.tight_layout()
