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

input_root_rawdata = r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\BIDS_Berlin_ECOG_LFP\rawdata_new"
bids_root = input_root_rawdata
vhdr_paths = get_all_paths(input_root_rawdata,'.vhdr')

excelfile=r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\_non_Datasets\Documents\BIDS_Berlin_LFP_ECOG_templates\BIDS_Berlin_conversion_20210420.xls"
#make list of dictionary with list index the row number, and the dictionary has key:value pair as columnname:data
list_of_dicts= xl2dict.XlToDict()
list_of_dicts=list_of_dicts.convert_sheet_to_dict(file_path=excelfile)

print(*vhdr_paths, sep="\n")


for vhdr_path in vhdr_paths:
    vhdr_file=os.path.basename(vhdr_path)
    searchfor=vhdr_file[:-10]
    current_row_dict = next((item for item in list_of_dicts if item["Filename_BIDS"] == searchfor), None)
    print(searchfor)
    print(vhdr_file)
    print(vhdr_path)
    bids_file = current_row_dict["Filename_BIDS"]

    entities = mne_bids.get_entities_from_fname(bids_file)

    bids_path = mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], task=entities["task"], \
                                      run=entities["run"], acquisition=entities["acquisition"], datatype="ieeg",
                                      root=bids_root)  # , suffix="ieeg")

    bids_path=bids_path.copy().update(extension='.json', suffix='ieeg')
    entries=dict()
    print(bids_path.basename)
    print(bids_path.match())
    print(bids_path)

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
        TaskDescription="Selfpaced right wrist rotations performed on custom-built analog rotameter which translates degree of rotation to volt."
        Instructions="Perform 50 wrist rotations with your right hand with an interval of about 10 seconds. Do not count in between rotations."
    elif task == "SelfpacedRotationL":
        TaskName="Selfpaced Rotation L"
        TaskDescription = 'Selfpaced left wrist rotations performed on custom-built analog rotameter which translates degree of rotation to volt.'
        Instructions = 'Perform 50 wrist rotations with your left hand with an interval of about 10 seconds. Do not count in between rotations.'
    elif task == "BlockRotationR":
        TaskName="Block Rotation R"
        TaskDescription = 'Blocks of 30 seconds of rest followed by blocks of 30 seconds of continuous wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt. Performed with the right hand.'
        Instructions = 'Upon the auditory command "start", perform continuous wrist rotations with your right hand, until you perceive the auditory command "stop". Perform these wrist rotations as fast as possible and with the largest possible amplitude.'
    elif task == "BlockRotationL":
        TaskName="Block Rotation L"
        TaskDescription = 'Blocks of 30 seconds of rest followed by blocks of 30 seconds of continuous wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt. Performed with the left hand.'
        Instructions = 'Upon the auditory command "start", perform continuous wrist rotations with your left hand, until you perceive the auditory command "stop". Perform these wrist rotations as fast as possible and with the largest possible amplitude.'
    elif task == "BlockRotationWheel":
        TaskName="Block Rotation Wheel"
        TaskDescription = 'Blocks of 30 seconds of rest followed by blocks of 30 seconds of continuous rotation performed on a steering wheel which translates degree of rotation to volt.'
        Instructions = 'Upon the auditory command "start", perform continuous rotations with your, until you perceive the auditory command "stop".'
    elif task == "Evoked":
        TaskName="Evoked"
        TaskDescription = 'Evoked potentials recording. Single stimulation pulses of fixed amplitude following periods of high frequency stimulation with varying amplitude (0, 1.5 and 3 mA) per block.'
        Instructions = 'Do not move or speak and keep your eyes open.'
    elif task == "Rest":
        TaskName="Rest"
        TaskDescription = 'Rest recording'
        Instructions = 'Do not move or speak and keep your eyes open.'
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
        TaskDescription = 'Recording performed during part III of the UPDRS (Unified Parkinsons Disease Rating Scale) questionnaire.'
        Instructions = 'See UPDRS questionnaire.'
    elif task == "RestTransition":
        TaskName="Rest Transition"
        TaskDescription = 'Rest recording during transition from dopaminergic medication OFF to medication ON state'
        Instructions = 'Do not move or speak and keep your eyes open.'
    elif task == "SelfpacedSpeech":
        TaskName = "Selfpaced Speech"
        TaskDescription = 'Selfpaced reading aloud of the fable "The Parrot and the Cat" by Aesop. Extended pauses in between sentences.'
        Instructions = 'Read aloud sentence by sentence the text in front of you. Leave a pause of several seconds in between sentences.'
    elif task == "ReadRelaxMoveR":
        TaskName = "Read Relax Move R"
        TaskDescription = "Block of 30 seconds of continuous left wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt followed by a block of 30 seconds of rest followed by a block of 30 seconds of reading aloud (The Parrot and the Cat by Aesop). Multiple sets."
        Instructions = 'At the beginning of each block, a text will appear on the screen, specifying the task to be performed. An auditory cue will then be issued, marking the begin of your task. Perform the task until the next cue marks the end of the task. Tasks are either continuous right wrist rotation, resting with open eyes or reading aloud the text displayed on the screen.'
    elif task == "ReadRelaxMoveL":
        TaskName = "Read Relax Move L"
        TaskDescription = "Block of 30 seconds of continuous right wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt followed by a block of 30 seconds of rest followed by a block of 30 seconds of reading aloud ('The Parrot and the Cat' by Aesop). Multiple sets."
        Instructions = "At the beginning of each block, a text will appear on the screen, specifying the task to be performed. An auditory cue will then be issued, marking the begin of your task. Perform the task until the next cue marks the end of the task. Tasks are either continuous left wrist rotation, resting with open eyes or reading aloud the text displayed on the screen."
    elif task == "Visuomotor":
        TaskName="Visuomotor"
        TaskDescription = 'n/a'
        Instructions = 'n/a'
    elif task == "FingerTapping":
        TaskName="FingerTapping"
        TaskDescription = 'n/a'
        Instructions = 'n/a'



    DBS_electrode_manufacturer = current_row_dict["DBS_electrode_manufacturer"]
    if DBS_electrode_manufacturer == "Boston Directional":
        iEEGElectrodeGroups={"ECOG_strip": "1x6 AdTech strip on right sensorimotor cortex", "DBS_left": "1x8 Boston Scientific directional DBS lead (Cartesia) in left STN", "DBS_right": "1x8 Boston Scientific directional DBS lead (Cartesia) in right STN"}
    elif DBS_electrode_manufacturer == "Medtronic":
        iEEGElectrodeGroups={"ECOG_strip": "1x6 Ad-Tech strip on right sensorimotor cortex", "DBS_left": "1x4 Medtronic DBS lead in STN", "DBS_right": "1x4 Medtronic DBS lead in STN"}
    elif DBS_electrode_manufacturer == "Boston Directional SenSight":
        iEEGElectrodeGroups={"ECOG_strip": "1x6 Ad-Tech strip on right sensorimotor cortex", "DBS_left": "1x8 Medtronic directional DBS lead(SenSight) in left STN", "DBS_right": "1x8 Medtronic directional DBS lead(SenSight) in right STN."}


    if "StimOn" in entities["acquisition"]:
        ElectricalStimulation = True
    elif "StimOffOn" in entities["acquisition"]:
            ElectricalStimulation = True
    elif "StimOff" in entities["acquisition"]:
        ElectricalStimulation = False

    entries={
    'InstitutionName':'Charite - Universitaetsmedizin Berlin, corporate member of Freie Universitaet Berlin and Humboldt-Universitaet zu Berlin, Department of Neurology with Experimental Neurology/BNIC, Movement Disorders and Neuromodulation Unit',
    'InstitutionAddress': 'Chariteplatz 1, 10117 Berlin, Germany',
    'Manufacturer': Manufacturer,
    'ManufacturersModelName' : ManufacturersModelName,
    'TaskName': TaskName,
    'TaskDescription': TaskDescription,
    'Instructions' : Instructions,
    'iEEGReference': 'n/a',
    'SoftwareFilters' : 'n/a',
    'HardwareFilters' : 'n/a',
    'iEEGGround' : 'n/a',
    'iEEGPlacementScheme' : 'Right subdural cortical strip and bilateral subthalamic nucleus (STN) deep brain stimulation (DBS) leads.',
    'iEEGElectrodeGroups' : iEEGElectrodeGroups,
    'SubjectArtefactDescription' : 'n/a',
    'ElectricalStimulation' : ElectricalStimulation,
    'ElectricalStimulationParameters' : 'n/a'
    }

    mne_bids.update_sidecar_json(bids_path,entries)

for json in get_all_paths(bids_root, "scans.tsv"):
    # change source code to create json file if not exist: File "C:\Users\Jonathan\Documents\PYCHARM\Python\venv\lib\site-packages\mne_bids\sidecar_updates.py", line 95, in update_sidecar_json
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

        "medication_state" : {"Description": "state of medication during recording", "Levels": {
            "OFF": "OFF parkinsonian medication",
            "ON": "ON parkinsonian medication"}},
        "UPDRS-III" : {"Description": "Score of the unified Parkinson's disease rating scale (UPDRS), part III.",  "TermURL": "https://doi.org/10.1002/mds.10473"}
    }

    #"LED_ON" : {"LongName": "Levodopa equivalent dose", "Description": "dose of antiparkinsonian medication give in the experimental setup in between medication OFF and ON state expressed in an estimated equivalent L-Dopa dose", "Units": "milligram", "TermURL": "https://doi.org/10.1002/mds.23429"}

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
df = df.replace('nan', 'n/a',regex=True)
#Write dataframe into csv
df.to_csv(bids_root + os.sep + 'participants.tsv', sep='\t', encoding='utf-8',  index=False, line_terminator='\r\n', na_rep='n/a')
