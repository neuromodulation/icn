import os
import mne
import mne_bids
from bids import BIDSLayout
import numpy as np
from mayavi import mlab

os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

PATH_BERLIN = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Berlin_VoluntaryMovement"
annotations_out_Berlin = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Berlin_VoluntaryMovement\derivativess"

layout = BIDSLayout(PATH_BERLIN)
run_files_Berlin = layout.get(extension='.vhdr')

def read_BIDS_data(PATH_RUN, BIDS_PATH):
    """Given a run path and bids data path, read the respective data
    Parameters
    ----------
    PATH_RUN : string
    BIDS_PATH : string
    Returns
    -------
    raw_arr : mne.io.RawArray
    raw_arr_data : np.ndarray
    fs : int
    line_noise : int
    """
    entities = mne_bids.get_entities_from_fname(PATH_RUN)

    bids_path = mne_bids.BIDSPath(subject=entities["subject"],
                                  session=entities["session"],
                                  task=entities["task"],
                                  run=entities["run"],
                                  acquisition=entities["acquisition"],
                                  datatype="ieeg", root=BIDS_PATH)

    raw_arr = mne_bids.read_raw_bids(bids_path)

    return (raw_arr, raw_arr.get_data(), int(np.ceil(raw_arr.info["sfreq"])),
            int(raw_arr.info["line_freq"]))


if __name__ == '__main__':

    # 1. read raw data
    file_name = run_files_Berlin[0]
    raw_arr, data, sfreq, line_noise = read_BIDS_data(file_name, PATH_BEIJING)

    # 2. set annotations via visual inspection
    raw_arr.pick(['TTL_1_clean', 'ECOG_1_R_SM_HH', 'ECOG_2_R_SM_HH',
             'ECOG_3_R_SM_HH', 'ECOG_4_R_SM_HH', 'ECOG_5_R_SM_HH',
             'ECOG_6_R_SM_HH', 'ECOG_7_R_SM_HH', 'ECOG_8_R_SM_HH']).plot(scalings='auto')#, lowpass=80, highpass=5)

    # 3. save annotations in derivatives folder
    raw_arr.annotations.save(os.path.join(annotations_out_Beijing, file_name.filename[:-5]+'.txt'), overwrite=True)

    # now read data with annotations:

    # 1. read BIDS data again
    file_name = run_files_Berlin[0]
    raw_arr, data, sfreq, line_noise = read_BIDS_data(file_name, PATH_BEIJING)

    # 2. read annotations
    annot = mne.read_annotations(os.path.join(annotations_out_Beijing, file_name.filename[:-5]+".txt"))
    raw_arr.set_annotations(annot)

    # 3. potentially reject annotated data
    data = raw_arr.get_data(reject_by_annotation='omit')

    raw_arr.pick(['TTL_1_clean', 'ECOG_1_R_SM_HH', 'ECOG_2_R_SM_HH',
             'ECOG_3_R_SM_HH', 'ECOG_4_R_SM_HH', 'ECOG_5_R_SM_HH',
             'ECOG_6_R_SM_HH', 'ECOG_7_R_SM_HH', 'ECOG_8_R_SM_HH']).plot(scalings='auto')#, lowpass=80, highpass=5)
