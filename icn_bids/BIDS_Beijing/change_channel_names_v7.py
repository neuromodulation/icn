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

def preprocess_mov1(mov_dat):
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

def preprocess_mov2(mov_dat):
    # the TIME OFF in the TTL signal is ~50 ms
    MOV_ON = False
    mov_new = np.zeros(mov_dat.shape[0])
    mov_new[0] = mov_dat[0]
    mov_on_set = 0
    for i in range(mov_dat.shape[0]):
        if i > 0 and mov_dat[i] > 1:
            MOV_ON = True
            mov_on_set = i
        if (i - mov_on_set) > 700 and mov_dat[i] < 1:
            mov_new[i] = 0
            MOV_ON = False
        if MOV_ON is True:
            mov_new[i] = 1
    return mov_new


#sub-FOG011_ses-EphysMedOff_task-Rest_acq-StimOff_run-01_ieeg.vhdr
root=r"C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\conversion_room\changed_channelname_TTL1"
input=r"C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\conversion_room\write_with_pybv"
folder_dummy=r"C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\conversion_room"
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

    ######### first rename the channels
    #raw_arr.set_channel_types(mapping_channeltypes)
    mne.rename_channels(raw_arr.info, mapping=mapping_channelnames)

    ########### add clean TLL signal
    # first, add the fixed channel to the BIDS Data and create a new MNE RawArray using that
    data_ = raw_arr.get_data()
    print(data_.shape)


    # then I add for every TTL channel the cleaned version
    add_ = []
    add_label = []
    for ch_idx, ch in enumerate(raw_arr.ch_names):
        if "TTL" in ch:
            add_.append(preprocess_mov1(data_[ch_idx, :]))
            add_label.append("TTL_" + str(len(add_)) + "_clean")

    # stack the results to the previous brainvision data
    data_new = np.concatenate((data_, np.vstack(add_)), axis=0)
    ch_l = raw_arr.ch_names.copy()
    [ch_l.append(l) for l in add_label]  # updated channel list is now in ch_l

    pybv.write_brainvision(data=data_new,
                           sfreq=raw_arr.info["sfreq"],
                           ch_names=ch_l,
                           fname_base='dummy', folder_out=folder_dummy)

    # I will now read this file using mne io again
    raw_arr = mne.io.read_raw_brainvision(os.path.join(folder_dummy, 'dummy.vhdr'))


    ################### set channel types
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
    if bids_path.task == "ButtonPress":
        bids_path.update(task="ButtonPressL")
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


################## check the TTL channel at the last position
#root=r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\BIDS_Beijing_ECOG_LFP\rawdata"
vhdr_paths=get_all_vhdr_files(root)
for vhdr_path in vhdr_paths:
    filename=os.path.basename(vhdr_path)
    entities = mne_bids.get_entities_from_fname(vhdr_path)
    bids_path = mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], task=entities["task"], \
                                  run=entities["run"], acquisition=entities["acquisition"], datatype="ieeg",
                                  root=root)

    raw_arr = mne_bids.read_raw_bids(bids_path)


    try:
        start_sample=0
        stop_sample=50000
        channel_names = ["MISC_10_TTL","TTL_1_clean"]
        two_channels = raw_arr[channel_names, start_sample:stop_sample]
        x = two_channels[1]
        y = two_channels[0].T
        lines = plt.plot(x,y)
        plt.legend(lines, channel_names)
        plt.title(filename)
        #plt.plot(raw_arr.get_data(picks=["TTL_1_clean"], start=0, stop=50000))
        #plt.plot(raw_arr.get_data(picks=["MISC_10"], start=0, stop=50000))

    except:
        plt.plot(raw_arr.get_data()[1, 0:50000])
        plt.title(filename)

    plt.show()
