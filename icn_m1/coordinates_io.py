import numpy as np 
from bids import BIDSLayout
import os
import pandas as pd

class BIDS_coord:

    @staticmethod
    def get_coord_from_vhdr(BIDS_path, vhdr_file):
        
        subject = vhdr_file[vhdr_file.find('sub-')+4:vhdr_file.find('sub-')+7]
        dict_coord = {}
        
        if vhdr_file.find('right') !=-1:
            sess = 'right'
        else:
            sess = 'left'
        coord_path = os.path.join(BIDS_path, 'sub-'+ subject, 'ses-'+ sess, 'eeg', 'sub-'+ subject+ '_electrodes.tsv')
        df = pd.read_csv(coord_path, sep="\t")
        arr_name = list(df['name'])
        if np.array(df['name'].str.contains("ECOG")).any():
            dict_coord["coord_arr_vhdr_ECOG"] = np.ndarray.astype(np.array(df[df['name'].str.contains("ECOG")])[:, 1:4],
                                                        float).tolist()
        if np.array(df['name'].str.contains("STN")).any():
            dict_coord["coord_arr_vhdr_STN"] = np.ndarray.astype(np.array(df[df['name'].str.contains("STN")])[:, 1:4],float).tolist()
                                                                            
        return arr_name, dict_coord
            
    @staticmethod
    def read_BIDS_coordinates(BIDS_path):
        """
        from BIDS_path np array coordinate arrays are read and returned in list respective to subjects
        :return: coord_arr: array with shape (len(subjects), 4), where indexes in the following order: left ecog, left stn, right ecog, right stn
        :return: coord_arr_names: array with shape  (len(subjects), 4), where coord names are saved in order: left, right
        """
        layout = BIDSLayout(BIDS_path)
        subjects = layout.get_subjects()
        sessions = layout.get_sessions()
        coord_arr = np.empty((len(subjects), 4), dtype=object)  # left ecog, left stn, right ecog, right stn
        coord_arr_names = np.empty((len(subjects), 2), dtype=object)

        for subject_idx, subject in enumerate(subjects):
            for sess in sessions:

                #coord_path = BIDS_path + 'sub-' + subject + '/ses-' + sess + \
                #             '/eeg/sub-' + subject + '_electrodes.tsv'
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

                      