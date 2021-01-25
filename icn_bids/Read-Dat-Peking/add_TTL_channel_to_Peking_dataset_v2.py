# code adapted from https://github.com/neuromodulation/icn/blob/master/icn_bids/Read-Dat-Peking/Add%20TTL%20channels%20to%20Peking%20dataset.ipynb



import os
import scipy
import numpy as np
from matplotlib import pyplot as plt
from mne import io
#from bids import BIDSLayout
from mne.decoding import TimeFrequency
from matplotlib import pyplot as plt
from scipy import stats, signal
import mne
from mne import create_info, EpochsArray
from mne.time_frequency import tfr_morlet
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#cwd = os.getcwd()
#raw = mne.io.read_raw_edf('test.edf')

PATH = 'C:\\Users\\Jonathan\\Documents\\DATA\\PROJECT_Tiantan\\sourcedata'
cwd = os.getcwd()
def preprocess_mov(mov_dat):
    # the TIME OFF in the TTL signal is ~50 ms
    MOV_ON = False
    mov_new = np.zeros(mov_dat.shape[0])
    mov_new[0] = mov_dat[0]
    mov_on_set = 0
    for i in range(mov_dat.shape[0]):
        if i > 0 and mov_dat[i] > 1:
            MOV_ON = True
            mov_on_set = i
        if (i - mov_on_set) > 100 and mov_dat[i] < 1:
            mov_new[i] = 0
            MOV_ON = False
        if MOV_ON is True:
            mov_new[i] = 1
    return mov_new

# run through every patient and every run in the peking dataset and rewrite the TTL label

def get_all_edf_files(BIDS_path):
    """

    Given a BIDS path return all vhdr file paths without BIDS_Layout
    Args:
        BIDS_path (string)
    Returns:
        vhdr_files (list)
    """
    edf_files = []
    for root, dirs, files in os.walk(BIDS_path):
        for file in files:
            if file.endswith(".edf"):
                edf_files.append(os.path.join(root, file))
    return edf_files

edf_files = get_all_edf_files(PATH)

for edf_file in edf_files:
    print(edf_file)
    raw = mne.io.read_raw_edf(edf_file) # I do not know why Timon used to have an index with edf_files[8]
    #raw.save('test.edf')
    if "buttonPress" in edf_file:
        filename = os.path.basename(edf_file)
        info = mne.create_info(["POL DC10 clean"], raw.info["sfreq"], ch_types='emg')
        ch_clean = preprocess_mov(raw.get_data()[np.where(np.array(raw.ch_names) == "POL DC10")[0][0], :])
        raw_clean = mne.io.RawArray(np.expand_dims(ch_clean, axis=0), info)
        raw.add_channels([raw_clean.pick("POL DC10 clean")])
        fig = plt.figure()
        plt.plot(ch_clean)
        plt.axis([0, 2000000, 0, 1])

        plt.title(filename)
        #plt.show()
        fig.savefig("figure_POL_DC10_clean_"+filename)

        raw.save(filename) # this is the problem, it cannot save the file