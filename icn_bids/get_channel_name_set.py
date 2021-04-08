from pathlib import Path
from tempfile import NamedTemporaryFile
import pandas as pd
import os
import mne
import mne_bids
from mne_bids import write_raw_bids, BIDSPath

def get_all_paths(BIDS_path,extension):
    """

    Given a BIDS path return all vhdr file paths without BIDS_Layout
    Args:
        BIDS_path (string)
    Returns:
        vhdr_files (list)
    """
    paths = []
    for root, dirs, files in os.walk(BIDS_path):
        for file in files:
            if file.endswith(extension):
                paths.append(os.path.join(root, file))
    return paths


path=r"C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN\BERLIN_ECOG_LFP\rawdata"

my_channels = set()
my_path_files = get_all_paths(path, ".vhdr")
for my_path_file in my_path_files:
    raw=mne.io.read_raw_brainvision(my_path_file)
    for ch in raw.ch_names:
        if ch not in my_channels:
            my_channels.add(ch)

for elem in sorted(my_channels):
    print(elem)