import IO
import os
import mne_bids
from mne_bids import update_sidecar_json # append this to __init__.py of the mne bids module from https://mne.tools/mne-bids/stable/_modules/mne_bids/sidecar_updates.html#update_sidecar_json
import mne
import pandas
import json as js
from datetime import datetime
from datetime import timezone
import xl2dict
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings("ignore")


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

# Alternative way to make dictionaries in columns
#sheet=pandas.read_excel(r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\Documents\BIDS_Berlin_LFP_ECOG_templates\BIDS_Berlin_conversion_20200203 - Copy (8).xlsx")
#dictionary=dict(zip(sheet["File_BrainVision"],sheet["Filename_BIDS"]))
#dictionarydate=dict(zip(sheet["File_BrainVision"],sheet["acq_time"]))

input_root_rawdata=r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\Datasets\BIDS_Berlin\rawdata_Berlin"
input_root_sourcedata=r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\Datasets\BIDS_Berlin\sourcedata"
input_root_poly5=r"C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN\Poly5"
input_root_mpx_matlab=r"C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN\sourcedata_matlab\matlab_to_brainvision"
bids_root=r"C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN\BERLIN_ECOG_LFP_mpx_matlab"
excelfile=r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\Documents\BIDS_Berlin_LFP_ECOG_templates\BIDS_Berlin_conversion_20200203 - Copy (22).xlsx"
r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\Datasets\BIDS_Berlin\sourcedata"
#make list of dictionary with list index the row number, and the dictionary has key:value pair as columnname:data
list_of_dicts= xl2dict.XlToDict()
list_of_dicts=list_of_dicts.convert_sheet_to_dict(file_path=excelfile)

vhdr_paths = get_all_paths(input_root_sourcedata,'.vhdr')
vhdr_paths += get_all_paths(input_root_rawdata,'.vhdr')
poly5_paths = get_all_paths(input_root_poly5,'.vhdr') #get the poly5 to brainvision files converted by MATLAB Fieldtrip
mpx_and_matlab_paths = get_all_paths(input_root_mpx_matlab, '.vhdr')
vhdr_paths+=poly5_paths
vhdr_paths+=mpx_and_matlab_paths

# mpx_and_matlab_paths = get_all_paths(input_root_mpx_matlab, '.vhdr')
# vhdr_paths = mpx_and_matlab_paths
print(*vhdr_paths, sep="\n")





### events files
import shutil
event_paths = get_all_paths(input_root_rawdata,"events.tsv")
for event_path in event_paths:
    vhdr_file=os.path.basename(event_path)
    vhdr_file=vhdr_file.replace('_events.tsv', '_ieeg.vhdr')
    current_row_dict = next((item for item in list_of_dicts if item["File_BrainVision"] == vhdr_file), None)
    bids_file = current_row_dict["Filename_BIDS"]
    entities = mne_bids.get_entities_from_fname(bids_file)
    if entities["run"] == None:
        print("is being ignored because of run number:", vhdr_file, bids_file)

    elif entities["run"] != None:
        bids_path = mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], task=entities["task"], \
                                          run=entities["run"], acquisition=entities["acquisition"], datatype="ieeg",
                                          root=bids_root, suffix="events", extension='.tsv')
        shutil.copy(event_path, bids_root)
        os.rename(bids_root + os.sep + os.path.basename(event_path), bids_path.fpath )
        #mne_bids.write_raw_bids(vhdr_raw, bids_path, overwrite=True, verbose=False, events_data=event_path)












for vhdr_path in vhdr_paths:
    vhdr_file=os.path.basename(vhdr_path)

    #check whether poly5 is on location of MATLAB Fieldtrip poly5 conversion
    if vhdr_path in poly5_paths:
        poly5_file = os.path.splitext(vhdr_file)[0]+'.DATA.Poly5'

        current_row_dict = next((item for item in list_of_dicts if item["File_Original"] == poly5_file), None)
    elif vhdr_path in mpx_and_matlab_paths:
        mpx_matlab_file = vhdr_file
        current_row_dict = next((item for item in list_of_dicts if item["File_BrainVision"] == mpx_matlab_file), None)
    else:
        current_row_dict = next((item for item in list_of_dicts if item["File_BrainVision"] == vhdr_file), None)
    print(vhdr_file)
    print(vhdr_path)
    bids_file = current_row_dict["Filename_BIDS"]


    entities = mne_bids.get_entities_from_fname(bids_file)
    if entities["run"] == None:
        print("is being ignored because of run number:", vhdr_file, bids_file)

    elif entities["run"] != None:
        bids_path = mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], task=entities["task"], \
                                          run=entities["run"], acquisition=entities["acquisition"], datatype="ieeg",
                                          root=bids_root)#, suffix="ieeg")

        print("is being converted", vhdr_file, "\t => \t" , bids_file)
        vhdr_raw = mne.io.read_raw(vhdr_path, verbose=False)
        vhdr_raw.info['line_freq'] = 50
        set_chtypes=True
        if set_chtypes:
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

        try:
            dateobject=datetime.strptime(current_row_dict["acq_time"],'%Y-%m-%dT%H:%M:%S')
            #delta=timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=1, weeks=0)
            #dateobject=dateobject.astimezone(timezone.utc)
            dateobject = dateobject.replace(tzinfo=timezone.utc)
            #dateobject.utctimetuple()
            #dateobject=dateobject.replace(tzinfo=pytz.UTC)
            #timezone.utc
            #dateobject.utcoffset(delta)
            vhdr_raw.info['meas_date'] = dateobject# + 2 * delta
            #vhdr_raw.info['utc_offset'] = "+01:00"
        except:
            print("has no acquitision date: ", bids_file)

        # except:
        #     continue


        mne_bids.write_raw_bids(vhdr_raw, bids_path, overwrite=True, verbose=False)




        bids_path=bids_path.copy().update(extension='.json', suffix='ieeg')
        entries=dict()
        print(bids_path.basename)
        print(bids_path.match())
        source_extension=current_row_dict["source_extension"]
        if source_extension=='mpx':
            Manufacturer = "Alpha Omega Engineering Ltd. (AO)"
            ManufacturersModelName = "Neuro Omega"
        elif source_extension=='poly5':
            Manufacturer = "Twente Medical Systems Internationl B.V. (TMSi)"
            ManufacturersModelName = "Saga 64"
        elif source_extension=='vhdr':
            Manufacturer = "Brain Products GmbH"
            ManufacturersModelName = "n/a"
        elif source_extension=='smr':
            Manufacturer ="Cambridge Electronic Design (CED)"
            ManufacturersModelName = "n/a"

        task=entities["task"]
        if task == "SelfpacedRotationR":
            TaskName="Selfpaced Rotation R"
            TaskDescription="Selfpaced rotations performed on custom-built analog rotameter which translates degree of rotation to volt."
            Instructions="Perform 30 to 50 wrist rotations with your right hand with an interval of about 10 seconds. Do not count in between rotations."
        elif task == "SelfpacedRotationL":
            TaskName="Selfpaced Rotation L"
            TaskDescription = 'Selfpaced rotations performed on custom-built analog rotameter which translates degree of rotation to volt.'
            Instructions = 'Perform 30 to 50 wrist rotations with your left hand with an interval of about 10 seconds. Do not count in between rotations.'
        elif task == "BlockRotationR":
            TaskName="Block Rotation R"
            TaskDescription = 'n/a'
            Instructions = 'n/a'
        elif task == "BlockRotationL":
            TaskName="Block Rotation L"
            TaskDescription = 'n/a'
            Instructions = 'n/a'
        elif task == "BlockRotationWheel":
            TaskName="Block Rotation Wheel"
            TaskDescription = 'n/a'
            Instructions = 'n/a'
        elif task == "Evoked":
            TaskName="Evoked"
            TaskDescription = 'n/a'
            Instructions = 'n/a'
        elif task == "Rest":
            TaskName="Rest"
            TaskDescription = 'Rest recording'
            Instructions = 'n/a'
        elif task == "Speech":
            TaskName="Speech"
            TaskDescription = 'n/a'
            Instructions = 'n/a'
        elif task == "ContinuousStopping":
            TaskName="Continuous Stopping"
            TaskDescription = 'Performance of continuous circular forearm movements with a cursor on a screen using a digitizing tablet. Start, turn and stop events for transition from movement to rest are visually cued on screen after a randomized movement duration of 3–5 s and with randomized rest duration of 1–2 s before the go cue. Performance of at least 40 correct trials per condition.'
            Instructions = 'Continuously perform rotatory movements with your arm, in a comfortable movement speed, approximately within the grey circle shown onscreen. During movement, the circle can turn yellow. When it does, switch the direction of the rotatory arm movement as fast as possible. When the circle turns red, stop the rotatory movement as fast as possible. When the circle changes color from red to yellow, start the movement in the direction as demonstrated by the arrows shown above the circle.'
        elif task == "FreeDrawing":
            TaskName="Free Drawing"
            TaskDescription = 'n/a'
            Instructions = 'n/a'
        elif task == "UPDRSIII":
            TaskName="UPDRS-III"
            TaskDescription = 'n/a'
            Instructions = 'n/a'
        elif task == "RestTransition":
            TaskName="Rest Transition"
            TaskDescription = 'Rest recording during transition from dopaminergic medication OFF to medication ON state'
            Instructions = 'n/a'
        elif task == "Visuomotor":
            TaskName="Visuomotor"
            TaskDescription = 'n/a'
            Instructions = 'n/a'


        DBS_electrode_manufacturer = current_row_dict["DBS_electrode_manufacturer"]
        if DBS_electrode_manufacturer == "Boston Directional":
            iEEGElectrodeGroups={"ECOG_strip": "1x6 AdTech strip on left primary motor cortex", "DBS_left": "1x8 Boston Scientific directional DBS lead in STN", "DBS_right": "1x8 Boston Scientific directional DBS lead in STN"}
        elif DBS_electrode_manufacturer == "Medtronic":
            iEEGElectrodeGroups={"ECOG_strip": "1x6 AdTech strip on left primary motor cortex", "DBS_left": "1x4 Medtronic DBS lead in STN", "DBS_right": "1x4 Medtronic DBS lead in STN"}

        if "StimOn" in entities["acquisition"]:
            ElectricalStimulation=True
        elif "StimOff" in entities["acquisition"]:
            ElectricalStimulation = False

        entries={
        'InstitutionName':'Department of Neurology with Experimental Neurology, Universitaetsmedizin Charite',

        'InstitutionAddress': 'Chariteplatz 1, 10117 Berlin, Germany',
        'Manufacturer': Manufacturer,
        'ManufacturersModelName' : ManufacturersModelName,
        'TaskName': TaskName,
        'TaskDescription': TaskDescription,
        'Instructions' : Instructions,
        'iEEGReference': 'n/a',
        'SoftwareFilters' : 'n/a',
        'HardwareFilters' : 'HardwareFilters',
        'iEEGGround' : 'n/a',
        'iEEGPlacementScheme' : 'Left primary motor cortex electrocorticography (ECOG) strip and bilateral subthalamic nucleus (STN) deep brain stimulation (DBS) leads.',
        'iEEGElectrodeGroups' : iEEGElectrodeGroups,
        'SubjectArtefactDescription' : 'n/a',
        'ElectricalStimulation' : ElectricalStimulation,
        'ElectricalStimulationParameters' : 'n/a'
        }

        mne_bids.update_sidecar_json(bids_path,entries)









for json in get_all_paths(bids_root, "scans.tsv"):
    # change source code to create json file if not exist
    ###################################################
    # if not fpath.exists():
    #     with open(fpath, 'w') as outfile:
    #         json.dump(dict(),outfile)
    ###################################################
    # with open(json.split('.')[0] + ".json", "a") as outfile:
    #     js.dump(dict(), outfile)


    entities = mne_bids.get_entities_from_fname(json)
    bids_path=mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], suffix="scans", extension='.json',
                      root=bids_root)
    entries={
        "acq_time" : {"Description": "date of acquistion in the format YYYY-MM-DDThh:mm:ss", "Units": "datetime", "TermURL": "https://tools.ietf.org/html/rfc3339#section-5.6"},
        "LED_ON" : {"LongName": "Levodopa equivalent dose", "Description": "dose of antiparkinsonian medication give in the experimental setup in between medication OFF and ON state expressed in an estimated equivalent L-Dopa dose", "Units": "milligram", "TermURL": "https://doi.org/10.1002/mds.23429"}
    }
    mne_bids.update_sidecar_json(bids_path, entries)

    # change source code to create json file if not exist
    # with open(os.path.dirname(json) + os.sep + "ieeg" + os.sep + os.path.basename(json).split('.')[0] + "coordsystem.json", "a") as outfile:
    #     js.dump(dict(), outfile)
    bids_path = mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], extension='.json', datatype="ieeg", suffix="coordsystem",
                                  root=bids_root)


    entries = {

    'IntendedFor':'n/a',
    'iEEGCoordinateSystem':'Other',
    'iEEGCoordinateUnits':'mm',
    'iEEGCoordinateSystemDescription':'MNI',
    'iEEGCoordinateProcessingDescription':'n/a',
    'iEEGCoordinateProcessingReference':'n/a'

    }
    mne_bids.update_sidecar_json(bids_path, entries)

#add json files and data description to root:


# import ast
# ast.literal_eval "{{k['Keys']:}"
# #
list_of_dicts= xl2dict.XlToDict()
list_of_dicts=list_of_dicts.fetch_data_by_column_by_sheet_name(file_path=excelfile, sheet_name="dataset_description.json")
entries={k['Keys']: k['Values'] for k in list_of_dicts}
with open(bids_root+os.sep+"dataset_description.json","w") as fp:
    js.dump(entries, fp, indent=2)
#
bids_path=mne_bids.BIDSPath(root=bids_root, suffix="participants", extension='json')
list_of_dicts= xl2dict.XlToDict()
list_of_dicts=list_of_dicts.fetch_data_by_column_by_sheet_name(file_path=excelfile, sheet_name="participants.json")
entries={k['Keys']: k['Values'] for k in list_of_dicts}
mne_bids.update_sidecar_json(bids_path, entries)





#bids_path=mne_bids.BIDSPath(root=bids_root, suffix="participants", extension='tsv')

# rewrite the participants.tsv file with clinical data
import pandas as pd
#
# subject=None
#with open(bids_root + os.sep + 'participants.tsv', 'wt') as out_file:
#Read excel file into a dataframe
data_xlsx = pd.read_excel(excelfile, 'participants.tsv', index_col=None)

#Replace all columns having spaces with underscores
data_xlsx.columns = [c.replace(' ', '_') for c in data_xlsx.columns]

#Replace all fields having line breaks with space
df = data_xlsx.replace('\n', ' ',regex=True)
df = data_xlsx.replace('nan', 'n/a',regex=True)
#Write dataframe into csv
df.to_csv(bids_root + os.sep + 'participants.tsv', sep='\t', encoding='utf-8',  index=False, line_terminator='\r\n', na_rep='n/a')

