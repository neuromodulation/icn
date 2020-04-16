import json
from bids import BIDSLayout
import numpy as np
import os
import pandas as pd

class Settings:

    @staticmethod
    def load_JSON_settings(settings_name):
        """load and return the given JSON file
        
        Args:
            settings_name (string): absolute path of settings.json
        
        Returns:
            dict: settings parameter
        """
        with open(settings_name, 'r') as fp:
            setting_params = json.load(fp)
        return setting_params

    @staticmethod
    def get_DBS_patients(subject_path):
        """

        :param subject_path:
        :return: list with DBS folders in the given subject_path
        """
        list_DBS_folder = [i for i in os.listdir(subject_path) \
                           if i.startswith('DBS') and \
                           len([file for file in os.listdir(os.path.join(subject_path, i)) \
                                if file.startswith('stream_')]) != 0]
        return list_DBS_folder

    @staticmethod
    def read_all_vhdr_filenames(BIDS_path):
        """list of all vhdr file paths in BIDS_path
        
        Args:
            BIDS_path (string): absolute path of BIDS folder
        
        Returns:
            list: all vhdr file in given BIDS path
        """
        layout = BIDSLayout(BIDS_path)
        files = layout.get(extension='vhdr', return_type='filename')
        return files

    @staticmethod
    def read_BIDS_coordinates(BIDS_path):
        """from BIDS_path np array coordinate arrays are read and returned in list respective to subjects
        
        Args:
            BIDS_path (string): absolute BIDS path
        
        Returns:
            coord_arr (np array): array with shape (len(subjects), 4), where indexes in the following order: left ecog, left stn, right ecog, right stn,
            coord_arr_names (np array): array with shape  (len(subjects), 2), where coord names are saved in order: left, right
        """
        
        layout = BIDSLayout(BIDS_path)
        subjects = layout.get_subjects()
        sessions = layout.get_sessions()
        coord_arr = np.empty((len(subjects), 4), dtype=object)  # left ecog, left stn, right ecog, right stn
        coord_arr_names = np.empty((len(subjects), 2), dtype=object)

        for subject_idx, subject in enumerate(subjects):
            for sess in sessions:

                coord_path = os.path.join(BIDS_path, 'sub-'+ subject, 'ses-'+ sess, 'eeg', 'sub-'+ subject+ '_electrodes.tsv')
                
                print(coord_path)
                if os.path.exists(coord_path) is False:
                    continue
                df = pd.read_csv(coord_path, sep="\t")

                if sess == 'left':
                    if np.array(df['name'].str.contains("ECOG")).any():
                        coord_arr[subject_idx, 0] = np.ndarray.astype(np.array(df[df['name'].str.contains("ECOG")])[:, 1:4],
                                                                      float) # [1:4] due to bipolar referencing (first electrode missing)
                    if np.array(df['name'].str.contains("STN")).any():
                        coord_arr[subject_idx, 1] = np.ndarray.astype(np.array(df[df['name'].str.contains("STN")])[:, 1:4],
                                                                      float)
                    coord_arr_names[subject_idx, 0] = list(df['name'])
                elif sess == 'right':
                    if np.array(df['name'].str.contains("ECOG")).any():
                        coord_arr[subject_idx, 2] = np.ndarray.astype(np.array(df[df['name'].str.contains("ECOG")])[:, 1:4],
                                                                      float)
                    if np.array(df['name'].str.contains("STN")).any():
                        coord_arr[subject_idx, 3] = np.ndarray.astype(np.array(df[df['name'].str.contains("STN")])[:, 1:4],
                                                                      float)
                    coord_arr_names[subject_idx, 1] = list(df['name'])

        return coord_arr, coord_arr_names

    @staticmethod
    def define_grid():
        """plain hard coded definition of ECOG and STN grid, LEFT and RIGHT are here separated
        
        Returns:
            grid_left, grid_right, stn_left, stn_right (np array): respective grid arrays in shape (3, n_electroed)
        """
        grid_left = np.array([[-13.1000000000000, -35.5000000000000, -48.3000000000000, -60, -16.9000000000000,
                               -34.8000000000000, -67.5000000000000, -46.1000000000000, -59.8000000000000,
                               -14.2000000000000, -28.3000000000000, -42.3000000000000, -67.6000000000000,
                               -50.5000000000000, -14.6000000000000, -60.9000000000000, -31.6000000000000,
                               -5.10000000000000, -65.6000000000000, -41.8000000000000, -55.1000000000000,
                               -22.7000000000000, -5.80000000000000, -49.2000000000000, -34.5000000000000,
                               -61.5500000000000, -63.6000000000000, -40.4000000000000, -48.7000000000000,
                               -21.8000000000000, -58.2000000000000, -7, -36.3000000000000, -48.1000000000000,
                               -56.8000000000000, -7.30000000000000, -22.2000000000000, -36.8000000000000,
                               -46.8000000000000],
                              [-67.7000000000000, -60, -55.1000000000000, -51.8000000000000, -51.6000000000000,
                               -49.3000000000000, -47.1000000000000, -43.7000000000000, -39.6000000000000,
                               -39.1000000000000, -31.2000000000000, -30.7000000000000, -30.1000000000000,
                               -24.4000000000000, -22.7000000000000, -18.7000000000000, -16.9000000000000,
                               -12.6000000000000, -10.8000000000000, -10.2000000000000, -4.01000000000000, 1.20000000000000,
                               2.80000000000000, 3.70000000000000, 3.90000000000000, 6.20000000000000, 8.30000000000000,
                               11.8000000000000, 14.5000000000000, 16, 18.2000000000000, 18.4000000000000, 19.9000000000000,
                               24.6000000000000, 28.5200000000000, 33.8000000000000, 35, 35.4000000000000,
                               35.6000000000000],
                              [69.1000000000000, 66, 58.2000000000000, 48, 78, 71.7000000000000, 31, 61.1000000000000,
                               53.3000000000000, 81.1000000000000, 76, 70.2000000000000, 41.2000000000000, 64.4000000000000,
                               80.2000000000000, 50.9000000000000, 75.2000000000000, 77.3000000000000, 37.8000000000000, 67,
                               53.2000000000000, 72, 74.8000000000000, 54.7000000000000, 66.5000000000000, 35.9000000000000,
                               25.7000000000000, 60.7000000000000, 50.5000000000000, 68.9000000000000, 27.3000000000000,
                               70.3000000000000, 59.6000000000000, 44, 20.8000000000000, 61.7000000000000, 57.2000000000000,
                               47, 36]])
        stn_left = np.array([[-14.6, -13.2, -11.7, -9.10, -11.7, -13.2, -7.90, -10],
                             [-15.1, -15.1, -15.1, -12.6, -12.6, -12.6, -9.40, -10.1],
                             [-5.40, -7.20, -8.70, -8.70, -7.50, -5.10, -10.3, -7.80]])
        grid_right = np.copy(grid_left)
        grid_right[0, :] = grid_right[0, :] * -1
        stn_right = np.copy(stn_left)
        stn_right[0, :] = stn_right[0, :] * -1

        return grid_left, grid_right, stn_left, stn_right
    
    @staticmethod
    def write_coord_arr_to_list(coord_arr):
        """writes the given np coord_arr to a list a coordinates as a list of patients
        
        Args:
            coord_arr (np arr): 
        
        Returns:
            list: 
        """
        l_all = []
        for patient_idx in range(coord_arr.shape[0]):
            l_patient = []
            for idx_ in range(4):
                try:
                    l_patient.append(coord_arr[patient_idx, idx_].tolist())
                except:
                    l_patient.append(np.NaN)
            l_all.append(l_patient)
        return l_all

    @staticmethod
    def write_settings_json(BIDS_path):
        
        
        grid_left, grid_right, stn_left, stn_right = Settings.define_grid()

        coord_arr, coord_arr_names = read_BIDS_coordinates(BIDS_path)

        coord_arr_list = write_coord_arr_to_list(coord_arr)

        dict_ = {
            "subject_path" : '/Users/hi/Documents/lab_work/workshop_ML/subjects/', 
            "BIDS_path" : '/Users/hi/Documents/lab_work/workshop_ML/thesis_plots/BIDS/', 
            "out_path_folder" : '/Users/hi/Documents/lab_work/workshop_ML/thesis_plots/int_out/', 
            "out_path_folder_downsampled" : '/Users/hi/Documents/lab_work/workshop_ML/thesis_plots/int_out_downsampled/', 
            "sample_rate" : 1000, #read from BIDS
            "f_ranges" : [[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]], 
            "clip_low" : 3, 
            "clip_high" : 3, 
            "int_distance_ecog" : 20, 
            "int_distance_stn" : 10, 
            "line_noise" : 60, 
            "var_rolling_window" : 1, #ms
            "num_patients" : len(BIDSLayout(BIDS_path).get_subjects()),
            "list_DBS_folder" : [i for i in os.listdir(subject_path) \
                                if i.startswith('DBS') and \
                                len([file for file in os.listdir(os.path.join(subject_path, i)) \
                                        if file.startswith('stream_')]) != 0], 
            "vhdr_files" : layout.get(extension='vhdr', return_type='filename'),

            "grid_left" : grid_left.tolist(),
            "grid_right" : grid_right.tolist(),
            "grid_left" : grid_left.tolist(),
            "stn_right" : stn_right.tolist(),
            "stn_left" : stn_left.tolist(),
            "coord_arr" : coord_arr_list,  # left ecog, left stn, right ecog, right stn
            "coord_arr_names" : coord_arr_names.tolist()                           
    }


if __name__ == "__main__":
    setting_json_path = '/Users/hi/Documents/lab_work/icn/icn_m1/settings.json'
    Settings.load_JSON_settings(setting_json_path)