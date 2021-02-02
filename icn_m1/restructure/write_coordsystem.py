import os
from collections import OrderedDict

import numpy as np
import pandas as pd

import mne
import mne_bids

import IO

root = "/Users/richardkoehler/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - Data/Datasets/BIDS_Beijing/"
bids_out = "/Users/richardkoehler/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - Data/BIDS_conversionroom/BIDS_Beijing/tests/"

files = IO.get_all_files(path=root, suffix="edf", get_bids=True,
                         bids_root=root, verbose=True)

for file in files:
    bids_in = file
    raw = mne_bids.read_raw_bids(file)
    electr_file = None
    for f_name in os.listdir(bids_in.directory):
        print(f_name)
        if f_name.endswith('_electrodes.tsv'):
            electr_file = bids_in.directory / f_name

    if not electr_file:
        print('No file containing electrode coordinates found.')
    else:
        print('Electrode file being used: ', electr_file.name)
        df = pd.read_table(electr_file, sep='\t', header=None)
        data = df.values
        #data = np.genfromtxt(str(electr_file), dtype=str, delimiter='\t', comments=None, encoding='utf-8')
        column_names = data[0, :]
        info = data[1:, :]

        electrode_tsv = OrderedDict()
        for i, name in enumerate(column_names):
            electrode_tsv[name] = info[:, i].tolist()

        # Load in channel names
        ch_names = electrode_tsv['name']
        for idx, ch_name in enumerate(ch_names):
            ch_names[idx] = 'POL ' + ch_name

        # Load in the xyz coordinates as a float
        elec = np.empty(shape=(len(ch_names), 3))
        try:
            for ind, axis in enumerate(['x', 'y', 'z']):
                elec[:, ind] = list(map(float, electrode_tsv[axis]))
        except:
            try:
                for ind, axis in enumerate(['x_MNI', 'y_MNI', 'z_MNI']):
                    elec[:, ind] = list(map(float, electrode_tsv[axis]))
            except:
                print('No electrode coordinates found.')
        elec = elec / 1000  # convert mm to mne meter standard
        try:
            # Create mne montage
            montage = mne.channels.make_dig_montage(ch_pos=
                                                    dict(zip(ch_names, elec)),
                                                    coord_frame='mni_tal')
            # Set montage. Warning is issued if channels don't match. Consider getting locations for missing channels.
            raw.set_montage(montage, on_missing='warn', verbose=True)
        except:
            print('Montage was not possible.')
        mne_bids.write_raw_bids(raw, file.copy().update(root=bids_out), overwrite=False,
                                verbose=True)
