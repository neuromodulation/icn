# this code explains how to convert edf files to brainvision in a serialized way.
# code adapted from Timon
# from https://github.com/neuromodulation/icn/blob/master/icn_bids/Read-Dat-Peking/example_read_edf_write_bv.ipynb

import mne
from matplotlib import pyplot as plt
import mne_bids
import pybv
import os
from bids.layout import parse_file_entities

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

#where do you read in the data? BIDS like structure for edf
PATH = r'C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\sourcedata'
#where do you want to write your data?
root = r'C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\write_with_pybv'

edf_paths = get_all_edf_files(PATH)

for edf_path in edf_paths:
    raw = mne.io.read_raw_edf(edf_path)
    plt.plot(raw.get_data()[1,:])
    plt.show()
    dict = parse_file_entities(edf_path)
    subject = dict["subject"]
    session = dict["session"]
    run = dict["run"]
    task = dict["task"]
    #create new RawArray
    ieegdata = raw.get_data()
    info = mne.create_info(raw.ch_names, raw.info["sfreq"], ch_types='ecog')
    raw_new = mne.io.RawArray(ieegdata, info)
    bids_basename = mne_bids.BIDSPath(subject=subject, session=session, task=task, run=run, root=root)
    pybv.write_brainvision(data=ieegdata, sfreq=raw.info["sfreq"], ch_names=raw.ch_names, fname_base='dummy_write',
                           folder_out=root)
    bv_raw = mne.io.read_raw_brainvision(root + os.sep + 'dummy_write.vhdr')
    mapping = {}
    for ch in range(len(bv_raw.info['ch_names'])):
        mapping[bv_raw.info['ch_names'][ch]] = 'ecog'
    bv_raw.set_channel_types(mapping)
    bv_raw.info['line_freq'] = 50
    mne_bids.write_raw_bids(bv_raw, bids_path=bids_basename, overwrite=True)


    #  remove dummy file
    os.remove(root + os.sep +'dummy_write.vhdr')
    os.remove(root + os.sep +'dummy_write.eeg')
    os.remove(root + os.sep +'dummy_write.vmrk')
    new_vhdr = str(bids_basename.fpath)
    print(new_vhdr)
    raw_BV = mne.io.read_raw_brainvision(new_vhdr)
    plt.plot(raw.get_data()[1, :])
    plt.show()
