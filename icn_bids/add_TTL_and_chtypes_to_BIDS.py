import os
import mne
import mne_bids
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
def set_chtypes(vhdr_raw):
            #print('Setting new channel types...')
            remapping_dict = {}
            for ch_name in vhdr_raw.info['ch_names']:
                if ch_name.startswith('ECOG'):
                    remapping_dict[ch_name] = 'ecog'
                elif ch_name.startswith(('LFP', 'STN')):
                    remapping_dict[ch_name] = 'seeg'
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


path = r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\BIDS_conversionroom\BIDS_Pittsburgh_Gripforce\rawdata"
path = r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\BIDS_conversionroom\BIDS_Beijing\rawdata"
#outputpath= r"C:\Users\Jonathan\Documents\DATA\output"
bids_root=path

my_path_files = get_all_paths(path, ".vhdr")
for my_path_file in my_path_files:
    my_file = os.path.basename(my_path_file)
    entities = mne_bids.get_entities_from_fname(my_file)

    bids_path = mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], task=entities["task"],
                                          run=entities["run"], acquisition=entities["acquisition"], datatype="ieeg",
                                         extension='.vhdr', root=bids_root)
    raw = mne.io.read_raw_brainvision(my_path_file, preload=True)

    if "Button" in entities["task"]:
        print(my_path_file)



        matching = None
        matching = [s for s in raw.ch_names if "TTL" in s]
        if matching:
            TTL_channel_name = matching[0] #we only expect 1 ch type
            print(TTL_channel_name)
            MISCBUTTONPRESS = preprocess_mov(raw.get_data()[np.where(np.array(raw.ch_names) == matching)[0][0], :]) # gives a channel as output
            info = mne.create_info(["MISCBUTTONPRESS"], raw.info["sfreq"], ch_types='emg')
            raw_clean = mne.io.RawArray(np.expand_dims(MISCBUTTONPRESS, axis=0), info)
            #raw.add_channels([raw_clean])
            #raw_clean.load_data()
            raw.add_channels([raw_clean.pick("MISCBUTTONPRESS")])
            #raw.ch_names
            #raw.info
            #raw.chnames = raw.chnames + {MISC_BUTTONPRESS}

            #ieegdata = raw.get_data() # this is a numpy array
            #ieegdata


            #rawnew = mne.io.rawarray(, info)

    if True:
        print('before')
        for ch in range(len(raw.info['ch_names'])):
            print(raw.info['ch_names'][ch])
        print(raw.info)
        raw = set_chtypes(raw)
        print('after')
        print(raw.get_channel_types())
    raw.save(my_path_file, overwrite=True)
    raw = mne.io.read_raw_brainvision(my_path_file, preload=False)
    mne_bids.write_raw_bids(raw, bids_path, overwrite=True, verbose=False)
