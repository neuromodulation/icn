import os
import mne_bids
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
import pybv
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report

def get_all_vhdr_files(BIDS_path):
    """

    Given a BIDS path return all vhdr file paths without BIDS_Layout
    Args:
        BIDS_path (string)
    Returns:
        vhdr_files (list)
    """
    vhdr_files = []
    for root, dirs, files in os.walk(BIDS_path):
        for file in files:
            if file.endswith(".vhdr"):
                vhdr_files.append(os.path.join(root, file))
    return vhdr_files

root=r"C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\conversion_room\changed_channelname_new"
vhdr_paths=get_all_vhdr_files(root)
for vhdr_path in vhdr_paths:
    filename=os.path.basename(vhdr_path)
    entities = mne_bids.get_entities_from_fname(vhdr_path)
    bids_path = mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], task=entities["task"], \
                                  run=entities["run"], acquisition=entities["acquisition"], datatype="ieeg",
                                  root=root)

    raw_arr = mne_bids.read_raw_bids(bids_path)
    plt.plot(raw_arr.get_data()[1, :])
    plt.show()

print_dir_tree(root, max_depth=4)