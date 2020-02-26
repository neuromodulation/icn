import os
import numpy as np
import settings
import pickle
import settings

def get_int_runs(patient_idx):
    """

    :param patient_idx:
    :return: list with all run files for the given patient
    """
    os.listdir(settings.out_path_folder)
    if patient_idx < 10:
        subject_id = str('00') + str(patient_idx)
    else:
        subject_id = str('0') + str(patient_idx)
    list_subject = [i for i in os.listdir(settings.out_path_folder) if i.startswith('sub-'+subject_id) and i.endswith('.p')]
    return list_subject


def write_comb_patient(patient_idx):
    """
    This function rewrites the pickle files s.t. there is one file per patient only
    In this file all runs are grid point wise concatenated
    In a dict, for every grid point the concatenated data and the label stream gets written out
    BUT: this consumes a looot of space, that's why downsample_data() should be used
    :param patient_idx:
    :return:
    """
    print(patient_idx)
    dat_all = np.empty(94, dtype=object)

    runs_ = get_int_runs(patient_idx)
    act_ = np.zeros([94, len(runs_)])
    out_ = []
    for idx in range(len(runs_)):
        file = open(settings.out_path_folder + '/' + runs_[idx], 'rb')
        out = pickle.load(file)
        out_.append(out)
        act_[:, idx] = out['act_grid_points']

    for grid_point in range(94):
        start = 0

        if np.nonzero(act_[grid_point, :])[0].shape[0] == 0:
            continue

        for idx in np.nonzero(act_[grid_point, :])[0]:

            if start == 0:
                dat = out_[idx]['int_data'][grid_point]
                if grid_point < 39 or (grid_point > 78 and grid_point < 86):  # contralateral
                    label = out_[idx]['label_mov'][0, :]
                else:
                    label = out_[idx]['label_mov'][1, :]
                start = 1

            else:
                dat = np.concatenate((dat, out_[idx]['int_data'][grid_point]), axis=1)

                if grid_point < 39 or (grid_point > 78 and grid_point < 86):  # contralateral
                    label = np.concatenate((label, out_[idx]['label_mov'][0, :]), axis=0)
                else:
                    label = np.concatenate((label, out_[idx]['label_mov'][1, :]), axis=0)

        dat_all[grid_point] = {
            "dat": dat,
            "mov": label
        }
    arr_active_grid_points = np.zeros(94)
    arr_active_grid_points[np.nonzero(np.sum(act_, axis=1))[0]] = 1
    if patient_idx < 10:
        subject_id = str('00') + str(patient_idx)
    else:
        subject_id = str('0') + str(patient_idx)


    np.save(settings.out_path_folder + 'sub_' + subject_id + '_act_grid_points.npy', arr_active_grid_points)
    np.save(settings.out_path_folder + 'sub_' + subject_id + '_dat.npy', dat_all)

def downsample_data(downsample_rate = 10):
    """
    function reads every run file, downsamples it by the given factor, and rewrites it to different folder
    :return:
    """
    for patient_idx in range(16):
        print(patient_idx)
        runs_ = get_int_runs(patient_idx)
        for idx in range(len(runs_)):
            file = open(settings.out_path_folder + '/' + runs_[idx], 'rb')
            out = pickle.load(file)
            int_data = np.empty(94, dtype=object)
            label_mov =  np.empty(2, dtype=object)
            act_grid_points = out['act_grid_points']

            for grid_point in range(out['int_data'].shape[0]):
                if out['int_data'][grid_point] is None:
                    continue
                int_data[grid_point] = out['int_data'][grid_point][:,::downsample_rate]
            for lat_ in range(2):
                label_mov[lat_] =  out['label_mov'][lat_,::downsample_rate]
            label_ = np.zeros([2, label_mov[0].shape[0]])
            label_[0,:] = label_mov[0]
            label_[1,:] = label_mov[1]

            out = {
                "int_data": int_data,
                "label_mov": label_,
                "act_grid_points": act_grid_points
            }

            out_path_file = os.path.join(settings.out_path_folder, runs_[idx]) + '.p'
            pickle.dump(out, open(out_path_file, "wb"))


if __name__== "__main__":



    for patient_idx in range(16):
        write_comb_patient(patient_idx)