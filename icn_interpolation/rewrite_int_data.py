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
    list_subject = [i for i in os.listdir(settings.out_path_folder) if i.startswith('sub-'+subject_id)]
    return list_subject


# kriege die NaN Elemente dann trotzdem raus
folder_write_out = '/Users/hi/Documents/workshop_ML/thesis_plots/int_out/'


def write_comb_patient(patient_idx):
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