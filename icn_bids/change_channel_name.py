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

df = pd.read_excel(r'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN\ICN_CHANNEL_NAMING_apply.xlsx')


dictionary_channelnames = dict(zip(df["Original_Name"], df["CHANNEL NAME"]))
dictionary_channeltypes = dict(zip(df["Original_Name"], df["TYPE"]))


path=r"C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN\BERLIN_ECOG_LFP_mpx_matlab"

my_path_files = get_all_paths(path, ".vhdr")
for my_path_file in my_path_files:
    my_file = os.path.basename(my_path_file)
    entities = mne_bids.get_entities_from_fname(my_file)

    bids_path = mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], task=entities["task"],
                                          run=entities["run"], acquisition=entities["acquisition"], datatype="ieeg", suffix=entities["suffix"],
                                         extension='.vhdr', root=path)
    with NamedTemporaryFile(suffix='_raw.vhdr') as f:
        fname = f.name
        raw.save(fname, overwrite=True)
        raw = mne.io.read_raw_brainvision(my_path_file, preload=False)
        mapping_channelnames = {}
        mapping_channeltypes = {}
        for ch in range(len(raw.info['ch_names'])):
            mapping_channelnames[raw.info['ch_names'][ch]] = dictionary_channelnames[raw.info['ch_names'][ch]]
            mapping_channeltypes[raw.info['ch_names'][ch]] = dictionary_channeltypes[raw.info['ch_names'][ch]]

        # set all channel types to ECOG for iEEG - BIDS does not allow more than one channel type

        # raw_arr.set_channel_types(mapping_channeltypes)
        mne.rename_channels(raw.info, mapping=mapping_channelnames)
        mne.set_channeltype(raw.info, mapping=mapping_channelnames)

        write_raw_bids(raw=raw, bids_path=bids_path, overwrite=True)


