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
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
from bids import BIDSLayout

def set_chtypes(vhdr_raw):
    """
    define MNE RawArray channel types
    """
    #print('Setting new channel types...')
    remapping_dict = {}
    for ch_name in vhdr_raw.info['ch_names']:
        if ch_name.startswith('ECOG'):
            remapping_dict[ch_name] = 'ecog'
        elif ch_name.startswith(('LFP', 'STN')):
            remapping_dict[ch_name] = 'dbs'
        elif ch_name.startswith('EMG'):
            remapping_dict[ch_name] = 'emg'
        # mne_bids cannot handle both eeg and ieeg channel types in the same data
        elif ch_name.startswith('EEG'):
            remapping_dict[ch_name] = 'misc'
        elif ch_name.startswith(('MOV', 'ANALOG', 'ROT', 'ACC', 'AUX', 'X', 'Y', 'Z')):
            remapping_dict[ch_name] = 'misc'
        else:
            remapping_dict[ch_name] = 'misc'
    vhdr_raw.set_channel_types(remapping_dict, verbose=False)
    return vhdr_raw

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
#sub-FOG011_ses-EphysMedOff_task-Rest_acq-StimOff_run-01_ieeg.vhdr
root=r"C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\conversion_room\changed_channelname"
input=r"C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\conversion_room\write_with_pybv"
#input=r"C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\conversion_room\rawdata_Timon"
#dummy=r"C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\conversion_room\changed_channelname"
#root=r"C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\back_up\write_with_pybv"

df = pd.read_excel(r'C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\conversion_room'
                   r'\ICN_CHANNEL_NAMING_BEIJING.xlsx')

dictionary_channelnames = dict(zip(df["Original_Name"], df["CHANNEL NAME"]))
dictionary_channeltypes = dict(zip(df["Original_Name"], df["TYPE"]))

vhdr_paths=get_all_vhdr_files(input)
my_channels=set()

for vhdr_path in vhdr_paths:
    filename=os.path.basename(vhdr_path)
    #print(vhdr_path)
    #raw_BV = mne.io.read_raw_brainvision(vhdr_path)
    #print(raw_BV.get_data().shape)
    #print(raw_BV.ch_names)
    #print(raw_BV.info)

    entities = mne_bids.get_entities_from_fname(vhdr_path)
    bids_path = mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], task=entities["task"], \
                                  run=entities["run"], acquisition=entities["acquisition"], datatype="ieeg",
                                  root=input)

    raw_arr = mne_bids.read_raw_bids(bids_path)
    #print(raw_arr.info)
    #print(entities["subject"])
    #print(*raw_arr.info["ch_names"], sep="\n")
    my_channels=my_channels.union(set(raw_arr.info["ch_names"]))

    # set all channel types to ECOG for iEEG
    mapping_channelnames = {}
    mapping_channeltypes = {}

    for ch in range(len(raw_arr.info['ch_names'])):
        mapping_channelnames[raw_arr.info['ch_names'][ch]] = dictionary_channelnames[raw_arr.info['ch_names'][ch]]
    #    mapping_channeltypes[raw_arr.info['ch_names'][ch]] = dictionary_channeltypes[raw_arr.info['ch_names'][ch]]


    #raw_arr.set_channel_types(mapping_channeltypes)
    mne.rename_channels(raw_arr.info, mapping=mapping_channelnames)
    raw_arr=  set_chtypes( raw_arr)

    #### create empty BIDS electrode file ###########################################################
    # and the electrode.tsv file
    # using pybids I read the run files
    layout = BIDSLayout(root, validate=False)
    elec_file = layout.get(extension='tsv', suffix="electrodes", space="mni", subject=entities["subject"])
    # elec_file = elec_file.get_df()

    elec = np.empty(shape=(len(raw_arr.ch_names), 3))
    for ch_idx, ch in enumerate(raw_arr.ch_names):
        try:
            elec[ch_idx, :] = elec_file[elec_file["name"] == ch][["x", "y", "z"]]
        except:
            elec[ch_idx, :] = ["NaN", "NaN", "NaN"]

    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(raw_arr.ch_names, elec)), \
                                            coord_frame='mri')

    raw_arr.set_montage(montage, on_missing='warn')

    #pybv.write_brainvision(data=raw_arr.get_data(), sfreq=raw_arr.info["sfreq"], ch_names=raw_arr.ch_names,
    #fname_base=filename,folder_out=dummy)
    #bv_raw = mne.io.read_raw_brainvision(dummy + os.sep + 'dummy_write.vhdr')
    # mapping = {}
    # for ch in range(len(bv_raw.info['ch_names'])):
    #    mapping[bv_raw.info['ch_names'][ch]] = 'ecog'
    # bv_raw.set_channel_types(mapping)
    # bv_raw.info['line_freq'] = 50


    ###################################### CHANGE BIDS NAMING TO YOUR CONVENIENCE ################
    bids_path.update(root=root)
    bids_path.update(acquisition="StimOff")
    bids_path.update(session=''.join([entities["session"] + "01"]))


    mne_bids.write_raw_bids(raw_arr, bids_path=bids_path, overwrite=True)
    #
    #  # remove dummy file
    # os.remove(dummy + os.sep + 'dummy_write.vhdr')
    # os.remove(dummy + os.sep + 'dummy_write.eeg')
    # os.remove(dummy + os.sep + 'dummy_write.vmrk')
    # new_vhdr = str(bids_path.fpath)
    # print(new_vhdr)
    # raw_BV = mne.io.read_raw_brainvision(new_vhdr)
    # plt.plot(raw_arr.get_data()[1, :])
    # plt.show()


    #print(mapping)
    #for i in mapping:
    #    print(i, mapping[i], sep="\t")



print(mne_bids.print_dir_tree(bids_path, max_depth=4))

#print(my_channels)
print(*list(my_channels), sep="\n")
#print(*list(my_types), sep="\n")

# 45 channels in tsv file: "C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\conversion_room\rawdata\sub-FOG011\ses-EphysMedOff\ieeg\sub-FOG011_ses-EphysMedOff_task-Rest_acq-StimOff_run-01_channels.tsv"
#
#  --> ['POL RS1', 'POL RS2', 'POL RS3', 'POL RS4', 'POL RS5', 'POL RS6', 'POL RS7', 'POL RS8', 'POL RS9', 'POL RS10', 'POL RS13', 'POL RS14', 'POL RS15', 'POL RS16', 'POL RS17', 'POL RS18', 'POL RS19', 'POL RS20', 'POL RS21', 'POL E', 'POL RS22', 'POL RS23', 'POL RS11', 'POL RS12', 'POL RS24', 'POL RS25', 'POL RS26', 'POL RS27', 'POL RS28', 'POL RS29', 'POL RS30', 'POL LD1', 'POL LD2', 'POL LD3', 'POL LD4', 'POL RD1', 'POL RD2', 'POL DC10', 'POL DC11', 'POL RD3', 'POL RD4', 'POL $RS11', 'POL $RS12', 'POL LEMG1', 'POL LEMG2']
#
# 43 channels in raw file: "sub-FOG011_ses-EphysMedOff_task-Rest_acq-StimOff_run-01_ieeg.vhdr"
#
#  --> ['POL RS1', 'POL RS2', 'POL RS3', 'POL RS4', 'POL RS5', 'POL RS6', 'POL RS7', 'POL RS8', 'POL RS9', 'POL RS10', 'POL RS13', 'POL RS14', 'POL RS15', 'POL RS16', 'POL RS17', 'POL RS18', 'POL RS19', 'POL RS20', 'POL RS21', 'POL E', 'POL RS22', 'POL RS23', 'POL RS11', 'POL RS12', 'POL RS24', 'POL RS25', 'POL RS26', 'POL RS27', 'POL RS28', 'POL RS29', 'POL RS30', 'POL LD1', 'POL LD2', 'POL LD3', 'POL LD4', 'POL RD1', 'POL RD2', 'POL DC10', 'POL DC11', 'POL RD3', 'POL RD4', 'POL $RS11', 'POL $RS12']
#

