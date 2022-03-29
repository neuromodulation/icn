# This script provides a meta-json structure of a BIDS dataset, which can be afterwards modified and converted with Fieldtrip
# created by Jonathan Vanhoecke - ICN lab
# 21.02.2022

import os
import mne_bids
import json
from mne_bids import BIDSPath
from mne_bids.tsv_handler import _from_tsv
from mne_bids.path import _find_matching_sidecar

def get_all_vhdr_files(directory):
    """

    Given a BIDS path return all vhdr file paths without BIDS_Layout
    Args:
        BIDS_path (string)
    Returns:
        vhdr_files (list)
    """
    vhdr_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".vhdr"):
                vhdr_files.append(os.path.join(root, file))
    return vhdr_files


# Choose an output directory
os.chdir(r"C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_dev")

# where is the BIDS data located?
# root = r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\BIDS_Beijing_ECOG_LFP\rawdata"
# root=r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\BIDS_Berlin_ECOG_LFP\rawdata"
root = r"C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_dev\rawdata2"

import csv
header = ['file name', 'iEEGReference' , 'iEEGGround', 'ElectricalStimulation','LEFT','AnodalContact','CathodalContact','StimulationAmplitude','StimulationPulseWidth','StimulationFrequency','RIGHT','AnodalContactR','CathodalContactR','StimulationAmplitudeR','StimulationPulseWidthR','StimulationFrequencyR','SoftwareFilters','Manufacturer','HardwareFiltersUnipolarChannels','HardwareFiltersBipolarChannels','HardwareFiltersAuxiliaryChannels','AnalogueBandwidth','low_cutoff']
with open('data_overview.tsv', 'w', encoding='UTF8',newline='') as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    tsv_writer.writerow(header)

# iterate over the brainvision files
run_files = get_all_vhdr_files(root)
datatype = "ieeg"
for run_file in run_files:
    entities = mne_bids.get_entities_from_fname(run_file)
    entities["space"] = "MNI152NLin2009bAsym"
    bidspath = mne_bids.BIDSPath(
        subject=entities["subject"],
        session=entities["session"],
        task=entities["task"],
        run=entities["run"],
        acquisition=entities["acquisition"],
        datatype=datatype,
        root=root,
    )
    # raw_arr = mne_bids.read_raw_bids(bidspath)

    # read in associated subject info from participants.tsv
    participants_tsv_path = bidspath.root / "participants.tsv"
    participants_tsv = _from_tsv(participants_tsv_path)
    bids_participants = dict()
    for col_name, value in participants_tsv.items():
        subjects = participants_tsv["participant_id"]
        row_ind = subjects.index(f"sub-{bidspath.subject}")
        bids_participants[col_name] = value[row_ind]

    # read in scans_tsv
    scans_fname = BIDSPath(
        subject=bidspath.subject,
        session=bidspath.session,
        suffix="scans",
        extension=".tsv",
        root=bidspath.root,
    ).fpath
    scans_tsv = _from_tsv(scans_fname)
    bids_scans = dict()
    for col_name, value in scans_tsv.items():
        scans = scans_tsv["filename"]
        row_ind = scans.index(f"ieeg/{bidspath.basename}_ieeg.vhdr")
        bids_scans[col_name] = value[row_ind]

    # read in the electrodes tsv
    electrodes_fname = BIDSPath(
        subject=bidspath.subject,
        session=bidspath.session,
        datatype=datatype,
        space=entities["space"],
        suffix="electrodes",
        extension=".tsv",
        root=bidspath.root,
    ).fpath
    electrodes_tsv = _from_tsv(electrodes_fname)
    bids_electrodes = dict()
    for col_name, value in electrodes_tsv.items():
        bids_electrodes[col_name] = value

    # read the channels tsv
    channels_fname = BIDSPath(
        subject=bidspath.subject,
        session=bidspath.session,
        datatype=datatype,
        task=entities["task"],
        run=entities["run"],
        acquisition=entities["acquisition"],
        suffix="channels",
        extension=".tsv",
        root=bidspath.root,
    ).fpath
    channels_tsv = _from_tsv(channels_fname)
    bids_channels = dict()
    for col_name, value in channels_tsv.items():
        bids_channels[col_name] = value

    # read the ieeg json
    sidecar_fname = _find_matching_sidecar(bidspath, suffix=datatype, extension=".json")
    with open(sidecar_fname, "r", encoding="utf-8-sig") as fin:
        bids_sidecar_json = json.load(fin)

    # read the scans.json
    scans_json_fname = BIDSPath(
        subject=bidspath.subject,
        session=bidspath.session,
        suffix="scans",
        extension=".json",
        root=bidspath.root,
    ).fpath
    if scans_json_fname.exists():
        with open(scans_json_fname, "r", encoding="utf-8-sig") as fin:
            bids_scans_json = json.load(fin)

    # read the coords json
    coords_fname = BIDSPath(
        subject=bidspath.subject,
        session=bidspath.session,
        suffix="coordsystem",
        extension=".json",
        root=bidspath.root,
        space=entities['space'],
        datatype=datatype
    ).fpath
    with open(coords_fname, "r", encoding="utf-8-sig") as fin:
        bids_coords_json = json.load(fin)

    # now create an output json file will all this meta data
    bidsdict = dict()
    bidsdict["inputdata_location"] = run_file
    bidsdict["inputdata_fname"] = os.path.basename(run_file)
    bidsdict["entities"] = entities
    bidsdict["participants"] = bids_participants
    bidsdict["scans_tsv"] = bids_scans
    bidsdict["scans_json"] = bids_scans_json
    bidsdict["channels_tsv"] = bids_channels
    bidsdict["electrodes_tsv"] = bids_electrodes
    bidsdict["coord_json"] = bids_coords_json
    bidsdict["ieeg"] = bids_sidecar_json


    json_writeout = bidspath.basename + ".json"

    # with open(json_writeout, "w") as outfile:
    #     json.dump(bidsdict, outfile, indent=4)
    my_row = ['n/a']*23

    my_row[0] = bidsdict["inputdata_fname"]
    my_row[1] = bidsdict["ieeg"]["iEEGReference"]
    my_row[2] = bidsdict["ieeg"]["iEEGGround"]
    my_row[3] = bidsdict["ieeg"]["ElectricalStimulation"]
    try:
        my_row[5] = bidsdict["ieeg"]["ElectricalStimulationParameters"]["CurrentExperimentalSetting"]["Left"]["AnodalContact"]
        my_row[6] = bidsdict["ieeg"]["ElectricalStimulationParameters"]["CurrentExperimentalSetting"]["Left"]["CathodalContact"]
        my_row[7] = bidsdict["ieeg"]["ElectricalStimulationParameters"]["CurrentExperimentalSetting"]["Left"]["StimulationAmplitude"]
        my_row[8] = bidsdict["ieeg"]["ElectricalStimulationParameters"]["CurrentExperimentalSetting"]["Left"]["StimulationPulseWidth"]
        my_row[9] = bidsdict["ieeg"]["ElectricalStimulationParameters"]["CurrentExperimentalSetting"]["Left"]["StimulationFrequency"]
        my_row[4] = 'Left'
    except:
        pass
    try:
        my_row[11] = bidsdict["ieeg"]["ElectricalStimulationParameters"]["CurrentExperimentalSetting"]["Right"]["AnodalContact"]
        my_row[12] = bidsdict["ieeg"]["ElectricalStimulationParameters"]["CurrentExperimentalSetting"]["Right"]["CathodalContact"]
        my_row[13] = bidsdict["ieeg"]["ElectricalStimulationParameters"]["CurrentExperimentalSetting"]["Right"]["StimulationAmplitude"]
        my_row[14] = bidsdict["ieeg"]["ElectricalStimulationParameters"]["CurrentExperimentalSetting"]["Right"]["StimulationPulseWidth"]
        my_row[15] = bidsdict["ieeg"]["ElectricalStimulationParameters"]["CurrentExperimentalSetting"]["Right"]["StimulationFrequency"]
        my_row[10] = 'Right'
    except:
        pass
    my_row[16] = bidsdict["ieeg"]["SoftwareFilters"]
    my_row[17] = bidsdict["ieeg"]["Manufacturer"]
    try:
        my_row[18] = bidsdict["ieeg"]["HardwareFilters"]["Anti_AliasFilter"]["Low_Pass"]["UnipolarChannels"]
        my_row[19] = bidsdict["ieeg"]["HardwareFilters"]["Anti_AliasFilter"]["Low_Pass"]["BipolarChannels"]
        my_row[20] = bidsdict["ieeg"]["HardwareFilters"]["Anti_AliasFilter"]["Low_Pass"]["AuxiliaryChannels"]
        my_row[21] = bidsdict["ieeg"]["HardwareFilters"]["AnalogueBandwidth"]
    except:
        pass
    my_row[22] = '0'
    print(my_row)
    with open('data_overview.tsv', 'a+', encoding='UTF8', newline='\n') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        # write the data
        tsv_writer.writerow(my_row)
