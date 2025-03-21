# This script provides a meta-json structure of a BIDS dataset, which can be afterwards modified and converted with Fieldtrip
# created by Jonathan Vanhoecke - ICN lab
# 02.05.2023

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
os.chdir(r"C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_dev\metadata")

# where is the BIDS data located?
# root = r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\BIDS_Beijing_ECOG_LFP\rawdata"
root=r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\BIDS_01_Berlin_Neurophys\rawdata"
# root=r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\BIDS_01_Berlin_Neurophys\rawdata_27.04.2023"
# root = r"C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_dev\rawdata10c"

import csv
header = ['file name', 'current ref' , 'ground', 'stimulation','cathodal contact','amplitude','frequency','softwarefilter','manufacturer','hardware_high']
references = [];
run_file_list = [];
HardwareFiltersUnipolarChannels =[];
ElectricalStimulation = [];
#f = open("labbook.txt", "w")

need to remove the desc and need to update beh folder

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
        description=entities["description"],
        datatype=datatype,
        root=root,
    )
    print(bidspath)

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
        try:
            row_ind = scans.index(f"ieeg/{bidspath.basename}_ieeg.vhdr")
            bids_scans[col_name] = value[row_ind]
        except ValueError:
            error()
            if bidspath.description=='neurobehav' or bidspath.description=='behav':
                bids_scans['filename'] = f"ieeg/{bidspath.subject}_ieeg.vhdr"
                bidspath_with_neurophys_desc = bidspath.copy().update(description='neurophys')
                row_ind = scans.index(f"ieeg/{bidspath_with_neurophys_desc.basename}_ieeg.vhdr")
                bids_scans[col_name] = value[row_ind]

    # read the sessions tsv
    sessions_fname = BIDSPath(
        subject=bidspath.subject,
        suffix="sessions",
        extension=".tsv",
        root=bidspath.root,
    ).fpath

    sessions_tsv = _from_tsv(sessions_fname)
    bids_sessions = dict()
    sessions = sessions_tsv["session_id"]
    row_ind = sessions.index(f"ses-{bidspath.session}")
    for col_name, value in sessions_tsv.items():
        bids_sessions[col_name] = value[row_ind]

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
        description=entities["description"],
        suffix="channels",
        extension=".tsv",
        root=bidspath.root,
    ).fpath
    channels_tsv = _from_tsv(channels_fname)
    bids_channels = dict()
    for col_name, value in channels_tsv.items():
        bids_channels[col_name] = value


    # read the ieeg json
    sidecar_fname = _find_matching_sidecar(bidspath,suffix=datatype, extension=".json")
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

    # read the sessions.json
    sessions_json_fname = BIDSPath(
        subject=bidspath.subject,
        suffix="sessions",
        extension=".json",
        root=bidspath.root,
    ).fpath

    if sessions_json_fname.exists():
        with open(sessions_json_fname, "r", encoding="utf-8-sig") as fin:
            bids_sessions_json = json.load(fin)

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
#    bidsdict["scans_json"] = bids_scans_json
#    bidsdict["sesssions_json"] = bids_sessions_json
    bidsdict["channels_tsv"] = bids_channels
    bidsdict["electrodes_tsv"] = bids_electrodes
    bidsdict["coord_json"] = bids_coords_json
    bidsdict["ieeg"] = bids_sidecar_json
    bidsdict["sessions_tsv"] = bids_sessions


    json_writeout = bidspath.basename + ".json"

    with open(json_writeout, "w") as outfile:
        json.dump(bidsdict, outfile, indent=4)

    #run_file_list.append(bidsdict["inputdata_fname"])
    #references.append(bidsdict["ieeg"]["iEEGReference"])
    #try:
    #    HardwareFiltersUnipolarChannels.append(bidsdict["ieeg"]["HardwareFilters"]["Anti_AliasFilter"]["Low_Pass"]["UnipolarChannels"])
    #except:
    #    HardwareFiltersUnipolarChannels.append('n/a')
    #try:
    #    ElectricalStimulation.append(bidsdict["ieeg"]["ElectricalStimulation"])
    #except:
    #    ElectricalStimulation.append('n/a')


# print('\n'.join(run_file_list))
# print( '\n'.join(references))
# f = open("labbook.txt", "w")
# f.write('\n'.join(run_file_list))
# f.write('\n'.join(references))
# f.write('\n'.join(str(HardwareFiltersUnipolarChannels)))
# f.write('\n'.join(str(ElectricalStimulation)))
