# conda install -c conda-forge ipywidgets
# pip install ipywidgets
import ipywidgets as widgets
from IPython.display import display

# pip install ipyfilechooser
from ipyfilechooser import FileChooser
import os
import json
import re
from TMSiSDK.file_readers import Poly5Reader
import numpy as np
import mne
import ipympl
from ipywidgets import AppLayout
mne.viz.set_browser_backend('qt')  # Enable mne-qt-browser backend if mne < 1.0


# download from https://gitlab.com/tmsi/tmsi-python-interface/-/tree/main/TMSiSDK
# and do pip install https://gitlab.com/tmsi/tmsi-python-interface/-/blob/main/requirements_Windows.txt
style = {"description_width": "300px"}
layout = {"width": "800px"}

# initiate metadata dictionaries
metadict = {}
metadict['inputdata_location'] = str()
metadict['inputdata_fname'] = str()
metadict['entities'] = {}
metadict['entities']['subject'] = str()
metadict['entities']['session'] = str()
metadict['entities']['task'] = str()
metadict['entities']['acquisition'] = str()
metadict['entities']['run'] = str()
metadict['entities']['space'] = str()
metadict['participants'] = {}
metadict['participants']['participants_id'] = str()
metadict['participants']['sex'] = str()
metadict['participants']['handedness'] = str()
metadict['participants']['age'] = str()
metadict['participants']['date_of_implantation'] = str()
metadict['participants']['UPDRS_III_preop_OFF'] = str()
metadict['participants']['UPDRS_III_preop_ON'] = str()
metadict['participants']['disease_duration'] = str()
metadict['participants']['PD_subtype'] = str()
metadict['participants']['symptom_dominant_side'] = str()
metadict['participants']['LEDD'] = str()
metadict['participants']['DBS_target'] = str()
metadict['participants']['DBS_hemisphere'] = str()
metadict['participants']['DBS_manufacturer'] = str()
metadict['participants']['DBS_model'] = str()
metadict['participants']['DBS_directional'] = str()
metadict['participants']['DBS_contacts'] = str()
metadict['participants']['DBS_description'] = str()
metadict['participants']['ECOG_target'] = str()
metadict['participants']['ECOG_hemisphere'] = str()
metadict['participants']['ECOG_manufacturer'] = str()
metadict['participants']['ECOG_model'] = str()
metadict['participants']['ECOG_location'] = str()
metadict['participants']['ECOG_material'] = str()
metadict['participants']['ECOG_contacts'] = str()
metadict['participants']['ECOG_description'] = str()
metadict['scans_tsv'] = {}
metadict['scans_tsv']['filename'] = str()
metadict['scans_tsv']['acq_time'] = str()
metadict['sessions_tsv'] = {}
metadict['sessions_tsv']['acq_date'] = str()
metadict['sessions_tsv']['medication_state'] = str()
metadict['sessions_tsv']['UPDRS_III'] = str()
metadict['scans_json'] = {}
metadict['scans_json']['acq_time'] = {}
metadict['scans_json']['medication_state'] = {}
metadict['channels_tsv'] = {}
metadict['channels_tsv']['name'] = []
metadict['channels_tsv']['type'] = []
metadict['channels_tsv']['units'] = []
metadict['channels_tsv']['low_cutoff'] = []
metadict['channels_tsv']['high_cutoff'] = []
metadict['channels_tsv']['reference'] = []
metadict['channels_tsv']['group'] = []
metadict['channels_tsv']['sampling_frequency'] = []
metadict['channels_tsv']['notch'] = []
metadict['channels_tsv']['status'] = []
metadict['channels_tsv']['status_description'] = []
metadict['electrodes_tsv'] = {}
metadict['electrodes_tsv']['name'] = []
metadict['electrodes_tsv']['x'] = []
metadict['electrodes_tsv']['y'] = []
metadict['electrodes_tsv']['z'] = []
metadict['electrodes_tsv']['size'] = []
metadict['electrodes_tsv']['material'] = []
metadict['electrodes_tsv']['manufacturer'] = []
metadict['electrodes_tsv']['group'] = []
metadict['electrodes_tsv']['hemisphere'] = []
metadict['electrodes_tsv']['type'] = []
metadict['electrodes_tsv']['impedance'] = []
metadict['electrodes_tsv']['dimension'] = []
metadict['coord_json'] = {}
metadict['coord_json']['IntendedFor'] = str()
metadict['coord_json']['iEEGCoordinateSystem'] = str()
metadict['coord_json']['iEEGCoordinateUnits'] = str()
metadict['coord_json']['iEEGCoordinateSystemDescription'] = str()
metadict['coord_json']['iEEGCoordinateProcessingDescription'] = str()
metadict['coord_json']['iEEGCoordinateProcessingReference'] = str()
metadict['ieeg'] = {}
metadict['ieeg']['DeviceSerialNumber'] = str()
metadict['ieeg']['ECGChannelCount'] = int()
metadict['ieeg']['ECOGChannelCount'] = int()
metadict['ieeg']['EEGChannelCount'] = int()
metadict['ieeg']['EMGChannelCount'] = int()
metadict['ieeg']['EOGChannelCount'] = str()
metadict['ieeg']['ElectricalStimulation'] = bool()
metadict['ieeg']['HardwareFilters'] = str()
metadict['ieeg']['InstitutionAddress'] = str()
metadict['ieeg']['InstitutionName'] = str()
metadict['ieeg']['Instructions'] = str()
metadict['ieeg']['Manufacturer'] = str()
metadict['ieeg']['ManufacturersModelName'] = str()
metadict['ieeg']['MiscChannelCount'] = int()
metadict['ieeg']['PowerLineFrequency'] = int()
metadict['ieeg']['RecordingDuration'] = str()
metadict['ieeg']['RecordingType'] = str()
metadict['ieeg']['SEEGChannelCount'] = int()
metadict['ieeg']['SamplingFrequency'] = float()
metadict['ieeg']['SoftwareFilters'] = str()
metadict['ieeg']['SoftwareVersions'] = str()
metadict['ieeg']['TaskDescription'] = str()
metadict['ieeg']['TaskName'] = str()
metadict['ieeg']['TriggerChannelCount'] = int()
metadict['ieeg']['iEEGElectrodeGroups'] = str()
metadict['ieeg']['iEEGGround'] = str()
metadict['ieeg']['iEEGPlacementScheme'] = str()
metadict['ieeg']['iEEGReference'] = str()

bids_subject_prefix = widgets.RadioButtons(
    options=["EL", "L"],
    description="Subject Prefix:",
    style=style,
    layout=layout,
)
bids_subject = widgets.BoundedIntText(
    min=0, max=150, step=1, description="Subject nr:", style=style, layout=layout
)
bids_sex = widgets.Dropdown(
    options=["n/a", "female", "male", "other"],
    description="Sex:",
    style=style,
    layout=layout,
)
bids_handedness = widgets.RadioButtons(
    options=[
        "n/a",
        "right",
        "left",
        "equal"
    ],
    description="handedness",
    style=style,
    layout=layout,
)

bids_age = widgets.BoundedIntText(
    min=0, max=150, step=1, description="Age:", style=style, layout=layout
)
bids_date_of_implantation = widgets.DatePicker(
    description="Date of Implantation", style=style, layout=layout
)
bids_disease_duration = widgets.BoundedIntText(
    min=0, max=150, step=1, description="Disease duration:", style=style, layout=layout
)
bids_PD_subtype = widgets.RadioButtons(
    options=["n/a", "akinetic-rigid", "tremor-dominant", "equivalent"],
    description="PD subtype",
    style=style,
    layout=layout,
)
bids_symptom_dominant_side = widgets.RadioButtons(
    options=[
        "n/a",
        "right",
        "left",
        "equal"
    ],
    description="symptom dominant side",
    style=style,
    layout=layout,
)
bids_LEDD = widgets.BoundedIntText(
    max=10000,
    step=1,
    description="Levodopa equivalent daily dose (LEDD):",
    style=style,
    layout=layout,
)
bids_DBS_target = widgets.RadioButtons(
    options=["n/a", "STN", "GPI", "VIM"],
    description="DBS target",
    style=style,
    layout=layout,
)
bids_DBS_hemispheres = widgets.RadioButtons(
    options=["n/a", "right", "left", "bilateral"],
    description="DBS hemisphere",
    style=style,
    layout=layout,
)
bids_DBS_model = widgets.RadioButtons(
    options=[
        "n/a",
        "SenSight Short",
        "SenSight Long",
        "Vercise Cartesia X",
        "Vercise Cartesia",
        "Vercise Standard",
        "Abbott Directed Long",
        "Abbott Directed Short",
    ],
    description="DBS model",
    style=style,
    layout=layout,
)
bids_DBS_description = widgets.RadioButtons(
    options=[
        "n/a",
        "Medtronic: 8-contact, 4-level, directional DBS lead. 0.5 mm spacing.",
        "Medtronic: 8-contact, 4-level, directional DBS lead. 1.5 mm spacing.",
        "Boston Scientific: 16-contact, 5-level, directional DBS lead. 0.5 mm spacing.",
        "Boston Scientific: 8-contact, 4-level, directional DBS lead. 0.5 mm spacing.",
        "Boston Scientific: 8-contact, 8-level, non-directional DBS lead. 0.5 mm spacing.",
        "Abbott/St Jude: 8-contact, 4-level, directional DBS lead. 1.5 mm spacing.",
        "Abbott/St Jude: 8-contact, 4-level, directional DBS lead. 0.5 mm spacing.",
    ],
    description="DBS description",
    style=style,
    layout=layout,
)
mylink = widgets.jslink((bids_DBS_model, "index"), (bids_DBS_description, "index"))

ECOG_present = widgets.Button(
    value=False,
    description="ECOG present?",
    style=style,
    layout=layout,
)


def define_ECOG(click):
    with output1:
        ECOG_present.disabled = 1
        display(
            bids_ECOG_target,
            bids_ECOG_hemisphere,
            bids_ECOG_model,
            bids_ECOG_description,
        )


ECOG_present.on_click(define_ECOG)

bids_ECOG_target = widgets.RadioButtons(
    options=["n/a", "sensorimotor cortex"],
    description="ECOG target",
    style=style,
    layout=layout,
    value="n/a",
)
bids_ECOG_hemisphere = widgets.RadioButtons(
    options=[
        "n/a",
        "right",
        "left",
        "bilateral"
    ],
    description="ECOG hemisphere",
    style=style,
    layout=layout,
    value="n/a",
)
bids_ECOG_model = widgets.RadioButtons(
    options=[
        "n/a",
        "TS06R-AP10X-0W6",
        "DS12A-SP10X-000",
    ],
    description="ECOG model",
    style=style,
    layout=layout,
    value="n/a",
)
bids_ECOG_description = widgets.RadioButtons(
    options=[
        "n/a",
        "Ad-Tech: 6-contact, 1x6 narrow-body long term monitoring strip. Platinum contacts, 10mm spacing, contact size 4.0 mm diameter/1.8 mm exposure.",
        "Ad-Tech: 12-contact, 1x6 dual sided long term monitoring strip. Platinum contacts, 10mm spacing, contact size 4.0 mm diameter/2.3 mm exposure.",
    ],
    description="ECOG description",
    style=style,
    layout={"width": "max-content"},
    value="n/a",
)
mylink = widgets.jslink((bids_ECOG_model, "index"), (bids_ECOG_description, "index"))


output1 = widgets.Output()


##################################################################################################################################
##################################################################################################################################


session_creation = widgets.Button(
    description="Create a session", style=style, layout=layout
)


def on_session_creation(click1):
    with output2:

        bids_session.append(
            widgets.Text(
                description="Session name:",
                placeholder="Give in the session name",
                style=style,
                layout=layout,
            )
        )

        bids_UPDRS_session.append(
            widgets.Text(
                description="UPDRS session:",
                value="n/a",
                placeholder="Give in UPDRS score or 'n/a'",
                style=style,
                layout=layout,
            )
        )
        bids_space.append(
            widgets.Text(
                description="Space session:",
                value="MNI152NLin2009bAsym",
                placeholder="Give in the space",
                style=style,
                layout=layout,
            )
        )

        display(
            bids_session[-1],
            bids_UPDRS_session[-1],
            bids_space[-1],
            specify_file,
        )


session_creation.on_click(on_session_creation)
bids_session = []
bids_UPDRS_session = []
bids_space = []


specify_file = widgets.Button(
    description="specify file and specify recording-specific settings",
    style=style,
    layout=layout,
)

########################### to the recordings ##################################################
bids_filechooser = []
bids_task = []
bids_task_description = []
bids_task_instructions = []
bids_time_of_acquisition = []
bids_run = []
bids_acquisition = []
bids_channel_names_widgets = []
bids_channel_names_list = []
bids_reference = []
bids_status_description_widgets = []
bids_status_description_list = []
bids_stimulation_contact = []
bids_stimulation_amplitude_left = []
bids_stimulation_frequency_left = []
bids_stimulation_amplitude_right = []
bids_stimulation_frequency_right = []

task_options = [
    ("n/a", 0),
    ("Rest", 1),
    ("UPDRSIII", 2),
    ("SelfpacedRotationL", 3),
    ("SelfpacedRotationR", 4),
    ("BlockRotationL", 5),
    ("BlockRotationR", 6),
    ("Evoked", 7),
    ("SelfpacedSpeech", 8),
    ("ReadRelaxMoveR", 9),
    ("ReadRelaxMoveL", 10),
    ("VigorStimR", 11),
    ("VigorStimL", 12),
    ("SelfpacedHandTapL", 13),
    ("SelfpacedHandTapR", 14),
    ("SelfpacedHandTapB", 15),
    ("Free", 16),
]


def go_to_subsession(*args):
    global bids_channel_names_widgets
    global bids_channel_names_list
    global bids_reference
    global bids_status_description_widgets
    global bids_status_description_list
    global bids_stimulation_contact
    global bids_stimulation_amplitude_left
    global bids_stimulation_frequency_left
    global bids_stimulation_amplitude_right
    global bids_stimulation_frequency_right
    bids_channel_names_widgets = []
    bids_channel_names_list = []
    bids_reference = []
    bids_status_description_widgets = []
    bids_status_description_list = []
    bids_stimulation_contact = []
    def update_task(change):
        with output2:
            bids_task_description[-1].value = descriptions[change["new"]]
            bids_task_instructions[-1].value = instructions[change["new"]]

    bids_filechooser.append(FileChooser(os.getcwd()))
    bids_filechooser[-1].title = "selected POLY5 file to convert"

    bids_task.append(
        widgets.Dropdown(
            options=task_options,
            value=0,
            description="Task:",
            style=style,
            layout=layout,
        )
    )
    bids_task[-1].observe(update_task, names="value")

    descriptions = [
        "n/a",
        "Rest recording",
        "Recording performed during part III of the UPDRS (Unified Parkinson"
        "s Disease Rating Scale) questionnaire.",
        "Selfpaced left wrist rotations performed on custom-built analog rotameter which translates degree of rotation to volt.",
        "Selfpaced right wrist rotations performed on custom-built analog rotameter which translates degree of rotation to volt.",
        "Blocks of 30 seconds of rest followed by blocks of 30 seconds of continuous wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt. Performed with the left hand.",
        "Blocks of 30 seconds of rest followed by blocks of 30 seconds of continuous wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt. Performed with the right hand.",
        "Evoked potentials recording. Single stimulation pulses of fixed amplitude following periods of high frequency stimulation with varying amplitude (0, 1.5 and 3 mA) per block.",
        "Selfpaced reading aloud of the fable "
        "The Parrot and the Cat"
        " by Aesop. Extended pauses in between sentences.",
        "Block of 30 seconds of continuous right wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt followed by a block of 30 seconds of rest followed by a block of 30 seconds of reading aloud ("
        "The Parrot and the Cat"
        " by Aesop). Multiple sets.",
        "Block of 30 seconds of continuous left wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt followed by a block of 30 seconds of rest followed by a block of 30 seconds of reading aloud ("
        "The Parrot and the Cat"
        " by Aesop). Multiple sets.",
        "Performance of diagonal forearm movements with a cursor on a screen using a digitizing tablet. Start and stop events are visually cued on screen with a rest duration of 350 ms. 14 blocks with 32 movements each. In blocks 3-5/9-11 bilateral stimulation is applied for 300 ms if a movement is slower/faster than the previous two movements. The order of slow/fast blocks is alternated between participants.  Performed with the right hand.",
        "Performance of diagonal forearm movements with a cursor on a screen using a digitizing tablet. Start and stop events are visually cued on screen with a rest duration of 350 ms. 14 blocks with 32 movements each. In blocks 3-5/9-11 bilateral stimulation is applied for 300 ms if a movement is slower/faster than the previous two movements. The order of slow/fast blocks is alternated between participants.  Performed with the left hand.",
        "Selfpaced left hand tapping, circa every 10 seconds, without counting, in resting seated position.",
        "Selfpaced right hand tapping, circa every 10 seconds, without counting, in resting seated position.",
        "Bilateral selfpaced hand tapping in rested seated position, one tap every 10 seconds, the patient should not count the seconds. The hand should be raised while the wrist stays mounted on the leg. Correct the pacing of the taps when the tap-intervals are below 8 seconds, or above 12 seconds. Start with contralateral side compared to ECoG implantation-hemisfere. The investigator counts the number of taps and instructs the patients to switch tapping-side after 30 taps, for another 30 taps in the second side.",
        "Free period, no instructions, during Dyskinesia-Protocol still recorded to monitor the increasing Dopamine-Level",
    ]
    instructions = [
        "n/a",
        "Do not move or speak and keep your eyes open.",
        "See UPDRS questionnaire.",
        "Perform 50 wrist rotations with your left hand with an interval of about 10 seconds. Do not count in between rotations.",
        "Perform 50 wrist rotations with your right hand with an interval of about 10 seconds. Do not count in between rotations.",
        'Upon the auditory command "start", perform continuous wrist rotations with your left hand, until you perceive the auditory command "stop". Perform these wrist rotations as fast as possible and with the largest possible amplitude.',
        'Upon the auditory command "start", perform continuous wrist rotations with your right hand, until you perceive the auditory command "stop". Perform these wrist rotations as fast as possible and with the largest possible amplitude.',
        "Do not move or speak and keep your eyes open.",
        "Read aloud sentence by sentence the text in front of you. Leave a pause of several seconds in between sentences.",
        "At the beginning of each block, a text will appear on the screen, specifying the task to be performed. An auditory cue will then be issued, marking the begin of your task. Perform the task until the next cue marks the end of the task. Tasks are either continuous right wrist rotation, resting with open eyes or reading aloud the text displayed on the screen.",
        "At the beginning of each block, a text will appear on the screen, specifying the task to be performed. An auditory cue will then be issued, marking the begin of your task. Perform the task until the next cue marks the end of the task. Tasks are either continuous left wrist rotation, resting with open eyes or reading aloud the text displayed on the screen.",
        "Your task is to move your pen from one side of the screen to the other. When you see a square, please move your pen there and stay on the square until a new square appears on the other side. Then move the pen to the new square. Please move as fast as you can and touch the screen with the pen throughout the whole experiment.",
        "Your task is to move your pen from one side of the screen to the other. When you see a square, please move your pen there and stay on the square until a new square appears on the other side. Then move the pen to the new square. Please move as fast as you can and touch the screen with the pen throughout the whole experiment.",
        "Keep both hands resting on your legs, and tap with your left hand by raising the hand and fingers of your left hand, without letting the arm be lifted from the leg. Do not count in between rotations.",
        "Keep both hands resting on your legs, and tap with your right hand by raising the hand and fingers of your right hand, without letting the arm be lifted from the leg. Do not count in between rotations.",
        "Keep both hands resting on your legs. First tap with your left hand (if ECoG is implanted in the right hemisphere; if ECoG is implanted in left hemisphere, start with right hand) by raising the left hand and fingers while the wrist is mounted on the leg. Make one tap every +/- ten seconds. Do not count in between taps. After 30 taps, the recording investigator will instruct you to tap on with your right (i.e. left) hand. After 30 taps the recording investigator will instruct you to stop tapping.",
        "Free period, without instructions or restrictions, of rest between Rest-measurement and Task-measurements",
    ]

    bids_task_description.append(
        widgets.Textarea(
            value=descriptions[0],
            description="Task description:",
            style=style,
            layout=layout,
        )
    )
    bids_task_instructions.append(
        widgets.Textarea(
            value=instructions[0],
            description="Task instructions:",
            style=style,
            layout=layout,
        )
    )

    bids_time_of_acquisition.append(
        widgets.Text(
            description="Date and time of recording",
            placeholder="2022-01-24T09:36:27 (use this format)",
            style=style,
            layout=layout,
        )
    )

    bids_run.append(
        widgets.BoundedIntText(
            value=1,
            min=0,
            max=10,
            step=1,
            description="Run number:",
            style=style,
            layout=layout,
        )
    )

    bids_acquisition.append(
        widgets.Text(
            description="acquisition e.g. StimOff StimOnL or StimOnBDopa30",
            style=style,
            layout=layout,
        )
    )

    with output2:

        display(
            bids_filechooser[-1],
            bids_task[-1],
            bids_task_description[-1],
            bids_task_instructions[-1],
            bids_time_of_acquisition[-1],
            bids_acquisition[-1],
            bids_run[-1],
            draw_channels,
        )



specify_file.on_click(go_to_subsession)
draw_channels = widgets.Button(
    description="go to channel plotting (wait long enough)",
    style=style,
    layout=layout,
)

go_to_reference = widgets.Button(
    description="define the reference and the channel status descriptions",
    style=style,
    layout=layout,
)

def plot_channels(*args):
    global bids_channel_names_widgets
    global bids_channel_names_list
    global bids_reference
    global bids_status_description_widgets
    global bids_status_description_list
    global bids_stimulation_contact
    global bids_stimulation_amplitude_left
    global bids_stimulation_frequency_left
    global bids_stimulation_amplitude_right
    global bids_stimulation_frequency_right
    bids_channel_names_widgets = []
    bids_channel_names_list = []
    bids_reference = []
    bids_status_description_widgets = []
    bids_status_description_list = []
    bids_stimulation_contact = []
    data = Poly5Reader(bids_filechooser[-1].selected_path + os.sep + bids_filechooser[-1].selected_filename)

    info = mne.create_info(ch_names=[ch._Channel__name for ch in data.channels], sfreq=data.sample_rate, ch_types=data.num_channels * ['misc'])

    raw = mne.io.RawArray(data.samples, info)


    for ch in raw.ch_names:

        if ('STN' in ch) or ('LFP' in ch) :
            preset = 'LFP_' + ch[3] + '_' + ''.join(filter(lambda i: i.isdigit(), ch)).rjust(2,"0") + '_'
            if 'STN' in ch: preset += 'STN_'
            if ch.endswith('B'):
                preset += 'BS'
            elif ch.endswith('M'):
                preset += 'MT'
            elif ch.endswith('MT'):
                preset += 'MT'
            elif ch.endswith('STN'):
                preset += 'MT'  # assume that Medtronic is standard
        elif ch.startswith('ECX'):
            if ch.startswith('ECXR10'):
                preset = 'ECOG_R_10_SMC_AT'
            elif ch.startswith('ECXR11'):
                preset = 'ECOG_R_11_SMC_AT'
            elif ch.startswith('ECXR12'):
                preset = 'ECOG_R_12_SMC_AT'
            else:
                if (ch[3] == 'R') and (bids_ECOG_hemisphere.value == 'right'):
                    ecog_side='R'
                elif (ch[3] == 'L') and (bids_ECOG_hemisphere.value == 'left'):
                    ecog_side = 'L'
                else:
                    ecog_side = 'HEMISPHERE_AMBIGUITY'
                preset = 'ECOG_' + ecog_side + '_' + ''.join(filter(lambda i: i.isdigit(), ch)).rjust(2,"0") + '_'
                if 'SM' in ch: preset += 'SMC_'
                if ch.endswith('B'):
                    preset += 'BS'
                elif ch.endswith('M'):
                    preset += 'MT'
                elif ch.endswith('A'):
                    preset += 'AT'
                elif ch.endswith('SMC'):
                    preset += 'AT' #assume that AT is standard
        elif ch.startswith('EEG'):
            preset = 'EEG_'
            if ch.upper().find('CZ')>0:
                preset += 'CZ_'
            if ch.upper().find('FZ')>0:
                preset += 'FZ_'
            if ch.upper().find('T')>0:
                preset += 'TM'
        elif re.search("R[0-9]C[0-9]",ch):
            preset = "EMG_L_" + re.search("R[0-9]C[0-9]",ch).group() + "_BR_TM"
        elif ch.startswith('BIP 01') or ch.startswith('EMGR'):
            preset = 'EMG_R_BR_TM'
        elif ch.startswith('BIP 02') or ch.startswith('EMGL'):
            preset = 'EMG_L_BR_TM'
        elif ch.startswith('BIP 03') or ch.startswith('ECG'):
            preset = 'ECG'
        elif ch.startswith('X-0'):
            preset = 'ACC_R_X_D2_TM'
        elif ch.startswith('Y-0'):
            preset = 'ACC_R_Y_D2_TM'
        elif ch.startswith('Z-0'):
            preset = 'ACC_R_Z_D2_TM'
        elif ch.startswith('X-1'):
            preset = 'ACC_L_X_D2_TM'
        elif ch.startswith('Y-1'):
            preset = 'ACC_L_Y_D2_TM'
        elif ch.startswith('Z-1'):
            preset = 'ACC_L_Z_D2_TM'
        elif ch.startswith('ISO aux') and (task_options[bids_task[-1].value][0] == 'SelfpacedRotationL' or task_options[bids_task[-1].value][0] == 'BlockRotationL' or task_options[bids_task[-1].value][0] == 'ReadRelaxMoveL'):
                preset = 'ANALOG_L_ROTA_CH'
        elif ch.startswith('ISO aux') and (task_options[bids_task[-1].value][0] == 'SelfpacedRotationR' or task_options[bids_task[-1].value][0] == 'BlockRotationR' or task_options[bids_task[-1].value][0] == 'ReadRelaxMoveR'):
                preset = 'ANALOG_R_ROTA_CH'
        else:
            preset = None

        channel_widget = widgets.Text(
            value=preset,
            placeholder='***deleted***',
            description=ch,
            style=style,
            layout=layout
        )
        bids_channel_names_widgets.append(channel_widget)



    with output2:
        #raw.plot(show=True, block=True, n_channels=raw.info['nchan'], title=bids_filechooser[-1].selected_filename)

        for widget in bids_channel_names_widgets:
            display(widget)

        raw.plot(show=True, block=False, n_channels=raw.info['nchan'], title=bids_filechooser[-1].selected_filename)

        display(go_to_reference)

draw_channels.on_click(plot_channels)

save_to_json = widgets.Button(
    description="Save this meta data to json and go to next recording",
    style=style,
    layout=layout,
)



def define_reference_and_stims(*args):
    global bids_stimulation_contact
    global bids_stimulation_amplitude_left
    global bids_stimulation_frequency_left
    global bids_stimulation_amplitude_right
    global bids_stimulation_frequency_right

    for widget in bids_channel_names_widgets:
        if widget.value != '':
            bids_channel_names_list.append(widget.value)
    bids_reference.append(
        widgets.Combobox(
        options=bids_channel_names_list,
        description='iEEG Reference: ',
        style=style,
        layout=layout
        )
    )
    for stimcon in range(0,8):
        bids_stimulation_contact.append(
            widgets.Combobox(
            options=bids_channel_names_list,
            description='Stimulation Contact: ',
            style=style,
            layout=layout
            )
        )
    bids_stimulation_frequency_left.append(widgets.BoundedIntText(
            value=130,
            min=1,
            max=1000,
            step=1,
            description='Stimulation Frequency left:',
            style=style,
            layout=layout
        ))

    bids_stimulation_frequency_right.append(widgets.BoundedIntText(
            value=130,
            min=1,
            max=1000,
            step=1,
            description='Stimulation Frequency right:',
            style=style,
            layout=layout
        ))


    bids_stimulation_amplitude_left.append( widgets.BoundedFloatText(
            value=0,
            min=0,
            max=5,
            step=0.1,
            description='Stimulation Amplitude left:',
            style=style,
            layout=layout
        ))


    bids_stimulation_amplitude_right.append(widgets.BoundedFloatText(
            value=0,
            min=0,
            max=5,
            step=0.1,
            description='Stimulation Amplitude right:',
            style=style,
            layout=layout
        ))




    with output2:
        display(bids_reference[-1])
        for stimcon in range(0,8):
            display(bids_stimulation_contact[stimcon])\

        display(bids_stimulation_amplitude_left[-1])
        display(bids_stimulation_amplitude_right[-1])
        display(bids_stimulation_frequency_left[-1])
        display(bids_stimulation_frequency_right[-1])
        display(go_to_status_description)


go_to_reference.on_click(define_reference_and_stims)
go_to_status_description = widgets.Button(
    description="define the channel status descriptions",
    style=style,
    layout=layout,
)

def define_status_description(*args):
    stimcontacts = []
    for stimcon in range(0,8):
        stimcontacts.append(bids_stimulation_contact[stimcon].value)
    for ch in bids_channel_names_list:
        if ch in stimcontacts:
            defaultvalue = 'Stimulation contact'
        elif ch in bids_reference[-1].value:
            defaultvalue = 'Reference electrode'
        else:
            defaultvalue = 'n/a'

        bids_status_description_widgets.append(
            widgets.Combobox(
            value=defaultvalue,
            options=['Reference electrode','Stimulation contact', 'Empty', 'Cable artefact'],
            description=ch,
            style=style,
            layout=layout
            )
        )
    with output2:
        display('status is assumed to be "good", and is always "bad" when specific description is given')

        for widget in bids_status_description_widgets:
            display(widget)
        display(save_to_json)

go_to_status_description.on_click(define_status_description)

def multiplefunctions_1(*args):
    go_to_subsession(*args)
    plot_channels(*args)

def multiplefunctions_2(*args):
    save_all_information(*args)
    with output2:
        display(session_creation)
        display(specify_file)

def save_all_information(*args):
    # All the vars that I want to get start with bids_

    metadict['inputdata_location'] = bids_filechooser[-1].selected_path + os.sep + bids_filechooser[-1].selected_filename
    metadict['inputdata_fname'] = bids_filechooser[-1].selected_filename
    metadict['entities'] = {}
    metadict['entities']['subject'] = str(bids_subject_prefix.value) + str(bids_subject.value).zfill(3)
    metadict['entities']['session'] = bids_session[-1].value
    metadict['entities']['task'] = task_options[bids_task[-1].value][0]
    metadict['entities']['acquisition'] = bids_acquisition[-1].value
    metadict['entities']['run'] = bids_run[-1].value
    metadict['entities']['space'] = bids_space[-1].value
    metadict['participants'] = {}
    metadict['participants']['participants_id'] = str()
    metadict['participants']['sex'] = bids_sex.value
    metadict['participants']['handedness'] = bids_handedness.value
    metadict['participants']['age'] = bids_age.value
    try:
        bids_date_of_implantation_str = bids_date_of_implantation.value
        metadict['participants']['date_of_implantation'] = bids_date_of_implantation_str.strftime("%Y-%m-%dT00:00:00")
    except:
        metadict['participants']['date_of_implantation'] = "unknown"
    finally:
        pass
    metadict['participants']['UPDRS_III_preop_OFF'] = str()
    metadict['participants']['UPDRS_III_preop_ON'] = str()
    metadict['participants']['disease_duration'] = bids_disease_duration.value
    metadict['participants']['PD_subtype'] = bids_PD_subtype.value
    metadict['participants']['symptom_dominant_side'] = bids_symptom_dominant_side.value
    metadict['participants']['LEDD'] = bids_LEDD.value
    metadict['participants']['DBS_target'] = bids_DBS_target.value
    metadict['participants']['DBS_hemisphere'] = bids_DBS_hemispheres.value
    metadict['participants']['DBS_manufacturer'] = str()
    metadict['participants']['DBS_model'] = bids_DBS_model.value
    metadict['participants']['DBS_directional'] = str()
    metadict['participants']['DBS_contacts'] = str()
    metadict['participants']['DBS_description'] = bids_DBS_description.value
    metadict['participants']['ECOG_target'] = bids_ECOG_target.value
    metadict['participants']['ECOG_hemisphere'] = bids_ECOG_hemisphere.value
    metadict['participants']['ECOG_manufacturer'] = str()
    metadict['participants']['ECOG_model'] = bids_ECOG_model.value
    metadict['participants']['ECOG_location'] = str()
    metadict['participants']['ECOG_material'] = str()
    metadict['participants']['ECOG_contacts'] = str()
    metadict['participants']['ECOG_description'] = bids_ECOG_description.value
    metadict['scans_tsv'] = {}
    metadict['scans_tsv']['filename'] = str()
    metadict['scans_tsv']['acq_time'] = bids_time_of_acquisition[-1].value
    metadict['sessions_tsv'] = {}
    metadict['sessions_tsv']['acq_date'] = bids_time_of_acquisition[-1].value[0:10]
    metadict['sessions_tsv']['medication_state'] = str()
    metadict['sessions_tsv']['UPDRS_III'] = bids_UPDRS_session[-1].value
    metadict['scans_json'] = {}
    metadict['scans_json']['acq_time'] = {}
    metadict['scans_json']['medication_state'] = {}
    metadict['channels_tsv']= {}
    metadict['channels_tsv']['name'] = bids_channel_names_list
    metadict['channels_tsv']['type'] = []
    metadict['channels_tsv']['units'] = []
    metadict['channels_tsv']['low_cutoff'] = []
    metadict['channels_tsv']['high_cutoff'] = []
    metadict['channels_tsv']['reference'] = []
    metadict['channels_tsv']['group'] = []
    metadict['channels_tsv']['sampling_frequency'] = []
    metadict['channels_tsv']['notch'] = []
    metadict['channels_tsv']['status'] = []
    metadict['channels_tsv']['status_description'] = []
    for widget in bids_status_description_widgets:
        if widget.value == 'n/a':
            metadict['channels_tsv']['status'].append('good')
        else:
            metadict['channels_tsv']['status'].append('bad')
        metadict['channels_tsv']['status_description'].append(widget.value)
    metadict['electrodes_tsv'] = {}
    metadict['electrodes_tsv']['name'] = []
    metadict['electrodes_tsv']['x'] = []
    metadict['electrodes_tsv']['y'] = []
    metadict['electrodes_tsv']['z'] = []
    metadict['electrodes_tsv']['size'] = []
    metadict['electrodes_tsv']['material'] = []
    metadict['electrodes_tsv']['manufacturer'] = []
    metadict['electrodes_tsv']['group'] = []
    metadict['electrodes_tsv']['hemisphere'] = []
    metadict['electrodes_tsv']['type'] = []
    metadict['electrodes_tsv']['impedance'] = []
    metadict['electrodes_tsv']['dimension'] = []
    metadict['coord_json'] = {}
    metadict['coord_json']['IntendedFor'] = str()
    metadict['coord_json']['iEEGCoordinateSystem'] = str()
    metadict['coord_json']['iEEGCoordinateUnits'] = str()
    metadict['coord_json']['iEEGCoordinateSystemDescription'] = str()
    metadict['coord_json']['iEEGCoordinateProcessingDescription'] = str()
    metadict['coord_json']['iEEGCoordinateProcessingReference'] = str()
    for stimcon in range(0,8):
        if bids_stimulation_contact[stimcon].value != "":
            if not 'stim' in metadict:
                metadict['stim'] ={}
                metadict['stim']['DateOfSetting'] = metadict['sessions_tsv']['acq_date']
                if bids_stimulation_amplitude_left[-1].value > 0:

                    metadict['stim']['L'] = {}
                    metadict['stim']['L']['CathodalContact'] = []
                    metadict['stim']['L']['StimulationAmplitude'] = bids_stimulation_amplitude_left[-1].value
                    metadict['stim']['L']['StimulationFrequency'] = bids_stimulation_frequency_left[-1].value
                if bids_stimulation_amplitude_right[-1].value > 0:
                    metadict['stim']['R'] = {}
                    metadict['stim']['R']['CathodalContact'] = []
                    metadict['stim']['R']['StimulationAmplitude'] = bids_stimulation_amplitude_right[-1].value
                    metadict['stim']['R']['StimulationFrequency'] = bids_stimulation_frequency_right[-1].value
            if '_L_' in bids_stimulation_contact[stimcon].value:
                metadict['stim']['L']['CathodalContact'].append(bids_stimulation_contact[stimcon].value)
            if '_R_' in bids_stimulation_contact[stimcon].value:
                metadict['stim']['R']['CathodalContact'].append(bids_stimulation_contact[stimcon].value)
    metadict['ieeg'] = {}
    metadict['ieeg']['DeviceSerialNumber'] = str()
    metadict['ieeg']['ECGChannelCount'] = int()
    metadict['ieeg']['ECOGChannelCount'] = int()
    metadict['ieeg']['EEGChannelCount'] = int()
    metadict['ieeg']['EMGChannelCount'] = int()
    metadict['ieeg']['EOGChannelCount'] = str()
    metadict['ieeg']['ElectricalStimulation'] = bool()
    metadict['ieeg']['HardwareFilters'] = str()
    metadict['ieeg']['InstitutionAddress'] = str()
    metadict['ieeg']['InstitutionName'] = str()
    metadict['ieeg']['Instructions'] = bids_task_instructions[-1].value
    metadict['ieeg']['Manufacturer'] = str()
    metadict['ieeg']['ManufacturersModelName'] = str()
    metadict['ieeg']['MiscChannelCount'] = int()
    metadict['ieeg']['PowerLineFrequency'] = int()
    metadict['ieeg']['RecordingDuration'] = str()
    metadict['ieeg']['RecordingType'] = str()
    metadict['ieeg']['SEEGChannelCount'] = int()
    metadict['ieeg']['SamplingFrequency'] = float()
    metadict['ieeg']['SoftwareFilters'] = str()
    metadict['ieeg']['SoftwareVersions'] = str()
    metadict['ieeg']['TaskDescription'] = str()
    metadict['ieeg']['TaskName'] = str()
    metadict['ieeg']['TriggerChannelCount'] = int()
    metadict['ieeg']['iEEGElectrodeGroups'] = str()
    metadict['ieeg']['iEEGGround'] = str()
    metadict['ieeg']['iEEGPlacementScheme'] = str()
    metadict['ieeg']['iEEGReference'] = bids_reference[-1].value
    metadict['poly5'] = {}
    metadict['poly5']['old'] = []
    metadict['poly5']['new'] = []
    for widget in bids_channel_names_widgets:
        metadict['poly5']['old'].append(widget.description)
        metadict['poly5']['new'].append(widget.value)

    currentfile = bids_filechooser[-1].selected_filename
    if not currentfile:
        with output2:
            print(currentfile)
            print("The information could not be saved, please select file below")

    else:
        currentfile += ".json"
        with open(currentfile, "w") as outfile:
            json.dump(metadict, outfile, indent=4)

        with output2:
            print("saving to: %.json", bids_filechooser[-1].selected_filename)
            print("information is saved and cannot be changed")



save_to_json.on_click(multiplefunctions_2)


output2 = widgets.Output()

# # overwrite previous list
#     bids_channel_names_widgets = []
#     bids_channel_names_list = []
#     bids_status_description_widgets = []
#     bids_status_description_list = []


BASE_MAPPING = {
    "LFPR1STNM": "LFP_R_01_STN_MT",
    "LFPR2STNM": "LFP_R_02_STN_MT",
    "LFPR3STNM": "LFP_R_03_STN_MT",
    "LFPR4STNM": "LFP_R_04_STN_MT",
    "LFPR5STNM": "LFP_R_05_STN_MT",
    "LFPR6STNM": "LFP_R_06_STN_MT",
    "LFPR7STNM": "LFP_R_07_STN_MT",
    "LFPR8STNM": "LFP_R_08_STN_MT",
    "LFPL1STNM": "LFP_L_01_STN_MT",
    "LFPL2STNM": "LFP_L_02_STN_MT",
    "LFPL3STNM": "LFP_L_03_STN_MT",
    "LFPL4STNM": "LFP_L_04_STN_MT",
    "LFPL5STNM": "LFP_L_05_STN_MT",
    "LFPL6STNM": "LFP_L_06_STN_MT",
    "LFPL7STNM": "LFP_L_07_STN_MT",
    "LFPL8STNM": "LFP_L_08_STN_MT",
    "LFPR01STN": "LFP_R_01_STN_MT",
    "LFPR02STN": "LFP_R_02_STN_MT",
    "LFPR03STN": "LFP_R_03_STN_MT",
    "LFPR04STN": "LFP_R_04_STN_MT",
    "LFPR05STN": "LFP_R_05_STN_MT",
    "LFPR06STN": "LFP_R_06_STN_MT",
    "LFPR07STN": "LFP_R_07_STN_MT",
    "LFPR08STN": "LFP_R_08_STN_MT",
    "LFPL01STN": "LFP_L_01_STN_MT",
    "LFPL02STN": "LFP_L_02_STN_MT",
    "LFPL03STN": "LFP_L_03_STN_MT",
    "LFPL04STN": "LFP_L_04_STN_MT",
    "LFPL05STN": "LFP_L_05_STN_MT",
    "LFPL06STN": "LFP_L_06_STN_MT",
    "LFPL07STN": "LFP_L_07_STN_MT",
    "LFPL08STN": "LFP_L_08_STN_MT",
    "EEGC1CzT": "EEG_CZ_TM",
    "EEGC1FzT": "EEG_FZ_TM",
    "EEGC1Cz_T": "EEG_CZ_TM",
    "EEGC2Fz_T": "EEG_FZ_TM",
    "EEGC1CzTM": "EEG_CZ_TM",
    "EEGC1FzTM": "EEG_FZ_TM",
    "ECXR1SMCA": "ECOG_R_01_SMC_AT",
    "ECXR2SMCA": "ECOG_R_02_SMC_AT",
    "ECXR3SMCA": "ECOG_R_03_SMC_AT",
    "ECXR4SMCA": "ECOG_R_04_SMC_AT",
    "ECXR5SMCA": "ECOG_R_05_SMC_AT",
    "ECXR6SMCA": "ECOG_R_06_SMC_AT",
    "ECXR01SMC": "ECOG_R_01_SMC_AT",
    "ECXR02SMC": "ECOG_R_02_SMC_AT",
    "ECXR03SMC": "ECOG_R_03_SMC_AT",
    "ECXR04SMC": "ECOG_R_04_SMC_AT",
    "ECXR05SMC": "ECOG_R_05_SMC_AT",
    "ECXR06SMC": "ECOG_R_06_SMC_AT",
    "R1C1": "EMG_L_R1C1_BR_TM",
    "R1C2": "EMG_L_R1C2_BR_TM",
    "R1C3": "EMG_L_R1C3_BR_TM",
    "R1C4": "EMG_L_R1C4_BR_TM",
    "R1C5": "EMG_L_R1C5_BR_TM",
    "R1C6": "EMG_L_R1C6_BR_TM",
    "R1C7": "EMG_L_R1C7_BR_TM",
    "R1C8": "EMG_L_R1C8_BR_TM",
    "R2C1": "EMG_L_R2C1_BR_TM",
    "R2C2": "EMG_L_R2C2_BR_TM",
    "R2C3": "EMG_L_R2C3_BR_TM",
    "R2C4": "EMG_L_R2C4_BR_TM",
    "R2C5": "EMG_L_R2C5_BR_TM",
    "R2C6": "EMG_L_R2C6_BR_TM",
    "R2C7": "EMG_L_R2C7_BR_TM",
    "R2C8": "EMG_L_R2C8_BR_TM",
    "R3C1": "EMG_L_R3C1_BR_TM",
    "R3C2": "EMG_L_R3C2_BR_TM",
    "R3C3": "EMG_L_R3C3_BR_TM",
    "R3C4": "EMG_L_R3C4_BR_TM",
    "R3C5": "EMG_L_R3C5_BR_TM",
    "R3C6": "EMG_L_R3C6_BR_TM",
    "R3C7": "EMG_L_R3C7_BR_TM",
    "R3C8": "EMG_L_R3C8_BR_TM",
    "R4C1": "EMG_L_R4C1_BR_TM",
    "R4C2": "EMG_L_R4C2_BR_TM",
    "R4C3": "EMG_L_R4C3_BR_TM",
    "R4C4": "EMG_L_R4C4_BR_TM",
    "R4C5": "EMG_L_R4C5_BR_TM",
    "R4C6": "EMG_L_R4C6_BR_TM",
    "R4C7": "EMG_L_R4C7_BR_TM",
    "R4C8": "EMG_L_R4C8_BR_TM",
    "BIP 01": "EMG_R_BR_TM",
    "BIP 02": "EMG_L_BR_TM",
    "BIP 03": "ECG",
    "X-0": "ACC_R_X_D2_TM",
    "Y-0": "ACC_R_Y_D2_TM",
    "Z-0": "ACC_R_Z_D2_TM",
    "X-1": "ACC_L_X_D2_TM",
    "Y-1": "ACC_L_Y_D2_TM",
    "Z-1": "ACC_L_Z_D2_TM",
    "ISO aux": f"ANALOG_{metadict['participants']['ECOG_hemisphere']}_ROTA_CH",
    }