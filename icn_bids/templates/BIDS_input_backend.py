# conda install -c conda-forge ipywidgets
# pip install ipywidgets
from typing import List, Any

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
    value="n/a",
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
    value="n/a"
)

bids_age = widgets.BoundedIntText(
    min=0, max=150, step=1, description="Age:", style=style, layout=layout, value=0,
)
'''
bids_date_of_implantation = widgets.DatePicker(
    description="Date of Implantation", style=style, layout=layout, value=
)
'''
bids_date_of_implantation = '2024-00-00T00:00:00'

bids_disease_duration = widgets.BoundedIntText(
    min=0, max=150, step=1, description="Disease duration:", style=style, layout=layout, value=99
)
bids_PD_subtype = widgets.RadioButtons(
    options=["n/a", "akinetic-rigid", "tremor-dominant", "equivalent"],
    description="PD subtype",
    style=style,
    layout=layout,
    value='akinetic-rigid',
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
    value="n/a",
)
'''
  bids_LEDD = widgets.BoundedIntText(
    max=10000,
    step=1,
    description="Levodopa equivalent daily dose (LEDD):",
    style=style,
    layout=layout,
)
'''
bids_DBS_target = widgets.RadioButtons(
    options=["n/a", "STN", "GPI", "VIM"],
    description="DBS target",
    style=style,
    layout=layout,
    value='STN',
)
bids_DBS_hemispheres = widgets.RadioButtons(
    options=["n/a", "right", "left", "bilateral"],
    description="DBS hemisphere",
    style=style,
    layout=layout,
    value='bilateral',
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
    value="SenSight Short",
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
    value="Medtronic: 8-contact, 4-level, directional DBS lead. 0.5 mm spacing.",
)
mylink = widgets.jslink((bids_DBS_model, "index"), (bids_DBS_description, "index"))

ECOG_present = widgets.Button(
    value=False,
    description="ECOG present?",
    style=style,
    layout=layout,
)

prefill_reference = widgets.Combobox(
        value='',
        options=['LFP_R_01_STN_MT','LFP_R_08_STN_MT','LFP_L_01_STN_MT','LFP_L_08_STN_MT'],
        description='iEEG pre-fill Reference: ',
        style=style,
        layout=layout
)

def define_ECOG(click):
    with output1:
        ECOG_present.disabled = 1
        bids_ECOG_target.value="sensorimotor cortex"
        bids_ECOG_model.value="TS06R-AP10X-0W6"
        bids_ECOG_description.value = "Ad-Tech: 6-contact, 1x6 narrow-body long term monitoring strip. Platinum contacts, 10mm spacing, contact size 4.0 mm diameter/1.8 mm exposure."
        display(
            #bids_ECOG_target,
            bids_ECOG_hemisphere,
            #bids_ECOG_model,
            #bids_ECOG_description,
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
        ses_options = ['EcogLfpMedOff01','EcogLfpMedOff02','EcogLfpMedOn01','EcogLfpMedOn02','EcogLfpMedOffOnDys01','EcogLfpMedOffOnDys02',
                       'LfpMedOff01', 'LfpMedOff02', 'LfpMedOn01', 'LfpMedOn02', 'LfpMedOffOnDys01', 'LfpMedOffOnDys02']

        bids_session.append(
            widgets.Combobox(
                options=ses_options,
                description="Session name:",
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
            #bids_UPDRS_session[-1],
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
bids_task_number = []
bids_time_of_acquisition = []
bids_run = []
bids_acquisition = []
bids_channel_names_widgets = []
bids_channel_names_list = []
bids_reference = []
bids_status_description_widgets = []
bids_status_description_list = []
bids_cathodal_contact = []
bids_stimulation_amplitude_left = 0
bids_stimulation_frequency_left = 0
bids_stimulation_amplitude_right = 0
bids_stimulation_frequency_right = 0
bids_stimulation_amplitude_min = 0
bids_stimulation_amplitude_max = 0
bids_stimulation_amplitude_stepsize = 0
bids_anodal_contact = []
hd_emg_muscle =[]

task_options = [
    ("n/a", 0),
    ("Rest", 1),
    ("UPDRSIII", 2),
    ("SelfpacedRotationL", 3),
    ("SelfpacedRotationR", 4),
    ("BlockRotationL", 5),
    ("BlockRotationR", 6),
    ("Evok", 7),
    ("EvokRamp", 8),
    ("SelfpacedSpeech", 9),
    ("ReadRelaxMoveR", 10),
    ("ReadRelaxMoveL", 11),
    ("VigorStimR", 12),
    ("VigorStimL", 13),
    ("VigorRailR", 14),
    ("VigorRailL", 15),
    ("SelfpacedHandTapL", 16),
    ("SelfpacedHandTapR", 17),
    ("SelfpacedHandTapB", 18),
    ("Free", 19),
    ("DyskinesiaProtocol",20),
    ("NaturalBehavior", 21),
    ("MotStopRailL", 22),
    ("MotStopRailR", 23),
]


def go_to_subsession(*args):
    global bids_channel_names_widgets
    global bids_channel_names_list
    global bids_reference
    global bids_status_description_widgets
    global bids_status_description_list
    global bids_cathodal_contact
    global bids_stimulation_amplitude_left
    global bids_stimulation_frequency_left
    global bids_stimulation_amplitude_right
    global bids_stimulation_frequency_right
    global bids_anodal_contact
    global hd_emg_muscle
    bids_channel_names_widgets = []
    bids_channel_names_list = []
    bids_reference = []
    bids_status_description_widgets = []
    bids_cathodal_contact= []
    bids_status_description_list=[]
    bids_stimulation_amplitude_left= 0
    bids_stimulation_frequency_left= 0
    bids_stimulation_amplitude_right= 0
    bids_stimulation_frequency_right= 0
    bids_anodal_contact = []
    def update_task(change):
        with output2:
            bids_task_description[-1].value = descriptions[change["new"]]
            bids_task_instructions[-1].value = instructions[change["new"]]

    FileChooser(os.getcwd())
    bids_filechooser.append(FileChooser(os.getcwd()))
    bids_filechooser[-1].title = "selected POLY5 file to convert"

    bids_task_number.append(
        widgets.Dropdown(
            options=task_options,
            value=0,
            description="Task:",
            style=style,
            layout=layout,
        )
    )
    bids_task_number[-1].observe(update_task, names="value")

    descriptions = [
        "n/a",
        "Rest recording",
        "Recording performed during part III of the UPDRS (Unified Parkinson"
        "s Disease Rating Scale) questionnaire.",
        "Selfpaced left wrist rotations performed on custom-built analog rotameter which translates degree of rotation to volt.",
        "Selfpaced right wrist rotations performed on custom-built analog rotameter which translates degree of rotation to volt.",
        "Blocks of 30 seconds of rest followed by blocks of 30 seconds of continuous wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt. Performed with the left hand.",
        "Blocks of 30 seconds of rest followed by blocks of 30 seconds of continuous wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt. Performed with the right hand.",
        'Evoked potentials recording. Single stimulation pulses of fixed amplitude following a block of a specified frequency.',
        'Evoked potentials recording. Single stimulation pulses of varying amplitude per block of a specified frequency.',
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
        "Performance of arm movements using the handle of a custom-made rail. The paradigm is structured into 4 blocks (Mov, Res, Con, Non) in pseudorandomized order. During each block the patient performs 12 movement episodes lasting 4 seconds. Each movement is followed by a break episode of the same length. During the movement episodes the patient is asked to move the handle from one side of the rail to the other as fast as possible. Movement and break episodes are cued on screen. 2 minutes break are forced between blocks. Stimulation during each block: Mov=Stimulation during movement segments (3-10), Res=Stimulation during break segments (3-10), Con=Stimulation during movement and break (3-10), Non=No stimulation.",
        "Performance of arm movements using the handle of a custom-made rail. The paradigm is structured into 4 blocks (Mov, Res, Con, Non) in pseudorandomized order. During each block the patient performs 12 movement episodes lasting 4 seconds. Each movement is followed by a break episode of the same length. During the movement episodes the patient is asked to move the handle from one side of the rail to the other as fast as possible. Movement and break episodes are cued on screen. 2 minutes break are forced between blocks. Stimulation during each block: Mov=Stimulation during movement segments (3-10), Res=Stimulation during break segments (3-10), Con=Stimulation during movement and break (3-10), Non=No stimulation.",
        "Selfpaced left hand tapping, circa every 10 seconds, without counting, in resting seated position.",
        "Selfpaced right hand tapping, circa every 10 seconds, without counting, in resting seated position.",
        "Bilateral selfpaced hand tapping in rested seated position, one tap every 10 seconds, the patient should not count the seconds. The hand should be raised while the wrist stays mounted on the leg. Correct the pacing of the taps when the tap-intervals are below 8 seconds, or above 12 seconds. Start with contralateral side compared to ECoG implantation-hemisfere. The investigator counts the number of taps and instructs the patients to switch tapping-side after 30 taps, for another 30 taps in the second side.",
        "Free period, no instructions, during Dyskinesia-Protocol still recorded to monitor the increasing Dopamine-Level",
        "Total concatenated recording of the dyskinesia protocol, as defined in the lab book",
        "Natural behavior observed when walking, chatting, drinking or eating at the hospital venue, outside of the experimental lab",
        "Performance of self-paced arm movements using the handle of a custom-made rail with the right hand. Stop instructions presented visually after decoding of movement intention.",
        "Performance of self-paced arm movements using the handle of a custom-made rail with the left hand. Stop instructions presented visually after decoding of movement intention.",
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
        "Do not move or speak and keep your eyes open.",
        "Read aloud sentence by sentence the text in front of you. Leave a pause of several seconds in between sentences.",
        "At the beginning of each block, a text will appear on the screen, specifying the task to be performed. An auditory cue will then be issued, marking the begin of your task. Perform the task until the next cue marks the end of the task. Tasks are either continuous right wrist rotation, resting with open eyes or reading aloud the text displayed on the screen.",
        "At the beginning of each block, a text will appear on the screen, specifying the task to be performed. An auditory cue will then be issued, marking the begin of your task. Perform the task until the next cue marks the end of the task. Tasks are either continuous left wrist rotation, resting with open eyes or reading aloud the text displayed on the screen.",
        "Your task is to move your pen from one side of the screen to the other. When you see a square, please move your pen there and stay on the square until a new square appears on the other side. Then move the pen to the new square. Please move as fast as you can and touch the screen with the pen throughout the whole experiment.",
        "Your task is to move your pen from one side of the screen to the other. When you see a square, please move your pen there and stay on the square until a new square appears on the other side. Then move the pen to the new square. Please move as fast as you can and touch the screen with the pen throughout the whole experiment.",
        "Your task is to move the handle from one side to the other as far and as fast as possible as soon as the Move-cue is given on the screen. As soon as you see the Break-cue you should stop moving the handle. Use right hand.",
        "Your task is to move the handle from one side to the other as far and as fast as possible as soon as the Move-cue is given on the screen. As soon as you see the Break-cue you should stop moving the handle. Use left hand.",
        "Keep both hands resting on your legs, and tap with your left hand by raising the hand and fingers of your left hand, without letting the arm be lifted from the leg. Do not count in between rotations.",
        "Keep both hands resting on your legs, and tap with your right hand by raising the hand and fingers of your right hand, without letting the arm be lifted from the leg. Do not count in between rotations.",
        "Keep both hands resting on your legs. First tap with your left hand (if ECoG is implanted in the right hemisphere; if ECoG is implanted in left hemisphere, start with right hand) by raising the left hand and fingers while the wrist is mounted on the leg. Make one tap every +/- ten seconds. Do not count in between taps. After 30 taps, the recording investigator will instruct you to tap on with your right (i.e. left) hand. After 30 taps the recording investigator will instruct you to stop tapping.",
        "Free period, without instructions or restrictions, of rest between Rest-measurement and Task-measurements",
        "Free period to be spend at the hospital venue (corridor, canteen, ...), without further instructions",
        "In this examination, you will perform movements with the handle in front of you. The handle is in the starting position when it is all the way to the left or all the way to the right. In the first phase, you will perform around 40 movements. Perform a single movement in each round. At the beginning of each round, return the handle to the starting position. When you have returned the handle to the starting position, a black cross will appear. Then stand ready for a movement. As soon as the cross appears, you must wait at least 1 second to perform the movement. However, you have up to 10 seconds. You decide when to make the move between 1 and 10 seconds after the cross appears. However, try NOT to count in your head. NOTE: DO NOT move the handle immediately after the cross appears. Wait at least 1 second and then decide for yourself when you want to move. From time to time, the word STOP will randomly appear in red as you move. Then you must react as quickly as possible and try to stop the movement. If you do not manage to stop the movement, it doesnot matter. Simply return the handle to the starting position for the next round.",
        "In this examination, you will perform movements with the handle in front of you. The handle is in the starting position when it is all the way to the left or all the way to the right. In the first phase, you will perform around 40 movements. Perform a single movement in each round. At the beginning of each round, return the handle to the starting position. When you have returned the handle to the starting position, a black cross will appear. Then stand ready for a movement. As soon as the cross appears, you must wait at least 1 second to perform the movement. However, you have up to 10 seconds. You decide when to make the move between 1 and 10 seconds after the cross appears. However, try NOT to count in your head. NOTE: DO NOT move the handle immediately after the cross appears. Wait at least 1 second and then decide for yourself when you want to move. From time to time, the word STOP will randomly appear in red as you move. Then you must react as quickly as possible and try to stop the movement. If you do not manage to stop the movement, it doesnot matter. Simply return the handle to the starting position for the next round.",
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
#    strdatetime = bids_filechooser[-1].selected_filename
#
#
#    bids_time_of_acquisition.append(
#        widgets.Text(
#            description="Date and time of recording",
#            value=strdatetime,
#            placeholder="2022-01-24T09:36:27 (use this format)",
#            style=style,
#            layout=layout,
#        )
#    )
    bids_run.append(
        widgets.BoundedIntText(
            value=1,
            min=0,
            max=100,
            step=1,
            description="Run number:",
            style=style,
            layout=layout,
        )
    )
    hd_emg_muscle.append(
        widgets.Dropdown(
            options=["BR","BB"],
            value="BR",
            description="hd-EMG muscle:",
            style=style,
            layout=layout,
        )
    )
    #bids_acquisition.append(
    #    widgets.Text(
    #        value=acq,
    #        description="acquisition e.g. StimOff StimOnL StimOnB StimOffDopa30",
    #        style=style,
    #        layout=layout,
    #    )
    #)
    with output2:

        display(
            bids_filechooser[-1],
            bids_task_number[-1],
            bids_task_description[-1],
            bids_task_instructions[-1],
            #bids_time_of_acquisition[-1],
            #bids_acquisition[-1],
            bids_run[-1],
            #hd_emg_muscle,
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
    global bids_cathodal_contact
    global bids_stimulation_amplitude_left
    global bids_stimulation_frequency_left
    global bids_stimulation_amplitude_right
    global bids_stimulation_frequency_right
    global bids_time_of_acquisition
    global bids_anodal_contact
    global bids_task
    bids_channel_names_widgets = []
    bids_channel_names_list = []
    bids_reference = []
    bids_status_description_widgets = []
    bids_cathodal_contact = []
    bids_stimulation_amplitude_left =0
    bids_stimulation_frequency_left =0
    bids_stimulation_amplitude_right =0
    bids_stimulation_frequency_right =0
    bids_time_of_acquisition = []
    bids_anodal_contact = []
    bids_task = task_options[bids_task_number[-1].value][0]
    with output2:
        display('Task is defined and stored')
    strdatetime = bids_filechooser[-1].selected_filename
    m = re.search(r'(20[0-9]{6}T[0-9]{6})', strdatetime)
    if m is not None:
        strdatetime = m.group(0)
        strdatetime = strdatetime[0:4] + '-' + strdatetime[4:6] + '-' + strdatetime[6:8] + strdatetime[8] + strdatetime[9:11] + ':' + strdatetime[11:13] + ':' + strdatetime[13:15]
    else:
        m = re.search(r'(202[0-9]{5}_[0-9]{6})', strdatetime)
        if m is not None:
            strdatetime = m.group(0)
            strdatetime = strdatetime[0:4] + '-' + strdatetime[4:6] + '-' + strdatetime[6:8] + 'T' + strdatetime[9:11] + ':' + strdatetime[11:13] + ':' + strdatetime[13:15]

    bids_time_of_acquisition.append(
        widgets.Text(
            description="Date and time of recording",
            value=strdatetime,
            placeholder="2022-01-24T09:36:27 (use this format)",
            style=style,
            layout=layout,
        )
    )

    stracq = bids_filechooser[-1].selected
    if 'StimOnBConMovNonRes' in stracq:
        stracq = 'StimOnBConMovNonRes'
    elif 'StimOnBNonResMovCon' in stracq:
        stracq = 'StimOnBNonResMovCon'
    elif 'StimOnBMovConResNon' in stracq:
        stracq = 'StimOnBMovConResNon'
    elif 'StimOnBResNonConMov' in stracq:
        stracq = 'StimOnBResNonConMov'
    elif 'EStimOnL' in stracq:
        stracq = 'EStimOnL'
    elif 'EStimOnR' in stracq:
        stracq = 'EStimOnR'
    elif 'StimOnL' in stracq:
        stracq = 'StimOnL'
    elif 'StimOnR' in stracq:
        stracq = 'StimOnR'
    elif 'StimOnB' in stracq:
        stracq = 'StimOnB'
    elif 'StimOn' in stracq:
        stracq = 'StimOn_SPECIFY_WHICH_SIDE'
    elif 'StimL' in stracq: #could be hand side instead of stim side
        stracq = 'StimOn_SPECIFY_WHICH_SIDE'
    elif 'StimR' in stracq:
        stracq = 'StimOn_SPECIFY_WHICH_SIDE'
    elif ('pre' or 'Pre') in stracq:
        stracq = 'StimOffDopaPre'
    elif 'Dopa' not in stracq:
        stracq = 'StimOff'
    else:
        m = re.search(r'(Dopa[0-9]{2,3})', stracq)
        if m is not None:
            stracq = 'StimOff' + m.group(0)
    if ('StimOff' in stracq) and ('Evok' in bids_task):
        with output2:
            display('Inconsistent acquisition!')
    if ('MedOn' in bids_session[-1].value) and ('MedOff' in bids_filechooser[-1].selected):
        with output2:
            display('Inconsistent session MedOff MedOn!')
    if ('MedOff' in bids_session[-1].value) and ('MedOn' in bids_filechooser[-1].selected):
        with output2:
            display('Inconsistent session MedOff MedOn!')

    bids_acquisition.append(
        widgets.Text(
            value=stracq,
            description="acquisition StimOff StimOnL StimOnB StimOffDopa30",
            style=style,
            layout=layout,
        )
    )

    data = Poly5Reader(bids_filechooser[-1].selected_path + os.sep + bids_filechooser[-1].selected_filename)

    info = mne.create_info(ch_names=[ch._Channel__name for ch in data.channels], sfreq=data.sample_rate, ch_types=data.num_channels * ['misc'])

    raw = mne.io.RawArray(data.samples, info)


    warning_EMG = False
    warning_SIDE = False

    for ch in raw.ch_names:

        if ('STN' in ch) or ('LFP' in ch) :
            preset = 'LFP_' + ch[3] + '_' + ''.join(filter(lambda i: i.isdigit(), ch)).rjust(2,"0") + '_'
            if 'STN' in ch: preset += 'STN_'
            if ch.endswith('BS') or 'Boston' in bids_DBS_description.value:
                preset += 'BS'
            elif ch.endswith('AB') or 'Abbott' in bids_DBS_description.value:
                preset += 'AB'
            elif ch.endswith('M'):
                preset += 'MT'
            elif ch.endswith('MT'):
                preset += 'MT'
            elif ch.endswith('STN'):
                preset += 'MT'  # assume that Medtronic is standard
        elif ch.startswith('ECXX'):
            ecog_side = bids_ECOG_hemisphere.value
            if ecog_side == 'right':
                ecog_side = 'R'
            elif ecog_side == 'left':
                ecog_side = 'L'
            preset = 'ECOG_' + ecog_side + '_' + ''.join(filter(lambda i: i.isdigit(), ch)).rjust(2, "0") + '_'
            if 'SM' in ch: preset += 'SMC_'
            if ch.endswith('B'):
                preset += 'BS'
            elif ch.endswith('M'):
                preset += 'MT'
            elif ch.endswith('A'):
                preset += 'AT'
            elif ch.endswith('SMC'):
                preset += 'AT'  # assume that AT is standard
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
                    warning_SIDE = True

                    if bids_ECOG_hemisphere.value == 'left':
                        ecog_side = 'L'
                    else:
                        ecog_side = 'R'
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
                preset += 'CZ_TM'
            if ch.upper().find('FZ')>0:
                preset += 'FZ_TM'
            #if ch.upper().find('T')>0:
            #    preset += 'TM'
        elif re.search("R[0-9]C[0-9]",ch):
            preset = "EMG_L_" + re.search("R[0-9]C[0-9]",ch).group() + "_" + hd_emg_muscle[-1].value + "_TM"
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
        elif ch.startswith('ISO aux') and (bids_task == 'SelfpacedRotationL' or bids_task == 'BlockRotationL' or bids_task == 'ReadRelaxMoveL'):
            preset = 'ANALOG_L_ROTA_CH'
        elif ch.startswith('ISO aux') and (bids_task == 'VigorRailL'):
            preset = 'RAIL_L'
        elif ch.startswith('ISO aux') and (bids_task == 'VigorRailR'):
            preset = 'RAIL_R'
        elif ch.startswith('TRIGGERS'):
            preset = 'TRIGGERS'
        elif ch in dictchannelnames:
            preset = dictchannelnames[ch]
            if preset.startswith('EMG_L') and bids_ECOG_hemisphere.value == 'left':
                warning_EMG = True
                if ecog_side =='L':
                    preset = preset.replace('EMG_L','EMG_R')
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
    if warning_SIDE:
        with output2:
            display('ECOG HEMISPHERE_AMBIGUITY_OR_CONFLICT_WITH_INPUT_ABOVE')
    if warning_EMG:
        with output2:
            print('warning, EMG is assumed to be contralateral to ecog side')



    with output2:
        #raw.plot(show=True, block=True, n_channels=raw.info['nchan'], title=bids_filechooser[-1].selected_filename)
        display(bids_time_of_acquisition[-1])
        display(bids_acquisition[-1])
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
    global bids_channel_names_widgets
    global bids_channel_names_list
    global bids_reference
    global bids_status_description_widgets
    global bids_status_description_list
    global bids_cathodal_contact
    global bids_stimulation_amplitude_left
    global bids_stimulation_frequency_left
    global bids_stimulation_amplitude_right
    global bids_stimulation_frequency_right
    global bids_anodal_contact
    global bids_stimulation_amplitude_min
    global bids_stimulation_amplitude_max
    global bids_stimulation_amplitude_stepsize

    bids_channel_names_list = []
    bids_reference = []
    bids_status_description_widgets = []
    bids_cathodal_contact = []
    bids_stimulation_amplitude_left = 0
    bids_stimulation_frequency_left = 0
    bids_stimulation_amplitude_right = 0
    bids_stimulation_frequency_right = 0
    bids_stimulation_amplitude_min = 0
    bids_stimulation_amplitude_max = 0
    bids_stimulation_amplitude_stepsize = 0

    for widget in bids_channel_names_widgets:
        if widget.value != '':
            bids_channel_names_list.append(widget.value)
    dropdown_ref_stim_contact = ['Ground']
    for ch in bids_channel_names_list:
        if 'ECOG' in ch or 'LFP' in ch:
            dropdown_ref_stim_contact.append(ch)

    bids_reference.append(
        widgets.Combobox(
        value=prefill_reference.value,
        options=dropdown_ref_stim_contact,
        description='iEEG Reference: ',
        style=style,
        layout=layout
        )
    )

    for anocon in range(0, 8):
        bids_anodal_contact.append(
            widgets.Combobox(
                options=dropdown_ref_stim_contact,
                description='Anodal Contact (+): ',
                style=style,
                layout=layout,
                value="",
            )
        )
    bids_cathodal_contact.append(
        widgets.Combobox(
            options=dropdown_ref_stim_contact,
            description='Cathodal Contact (-): ',
            style=style,
            layout=layout,
            value='Ground',
        )
    )
    for stimcon in range(1,8):
        bids_cathodal_contact.append(
            widgets.Combobox(
            options=dropdown_ref_stim_contact,
            description='Cathodal Contact (-): ',
            style=style,
            layout=layout,
            value="",
            )
        )



    if 'EStim' in bids_acquisition[-1].value:
        preset_freq = 5
    elif 'Evok' in bids_task:
        preset_freq = 5
    else:
        preset_freq = 130

    bids_stimulation_frequency_left=widgets.BoundedIntText(
            value=preset_freq,
            min=1,
            max=1000,
            step=1,
            description='Stimulation Frequency left:',
            style=style,
            layout=layout
        )

    bids_stimulation_frequency_right=widgets.BoundedIntText(
            value=preset_freq,
            min=1,
            max=1000,
            step=1,
            description='Stimulation Frequency right:',
            style=style,
            layout=layout
        )

    if 'EvokRamp' in bids_task:
        if 'StimOnL' in bids_acquisition[-1].value:
            preset_stimL = 5
            preset_stimR = 0
        elif 'StimOnR' in bids_acquisition[-1].value:
            preset_stimL = 0
            preset_stimR = 5
        elif 'StimOnB' in bids_acquisition[-1].value:
            preset_stimL = 5
            preset_stimR = 5
        else:
            with output2:
                print("ERROR EvokRamp has no side")

        bids_stimulation_amplitude_min = widgets.BoundedFloatText(
            value=0.5,
            min=0,
            max=10,
            step=0.1,
            description='Stimulation Amplitude minimum:',
            style=style,
            layout=layout
        )
        bids_stimulation_amplitude_max = widgets.BoundedFloatText(
            value=10,
            min=0,
            max=10,
            step=0.1,
            description='Stimulation Amplitude maximum:',
            style=style,
            layout=layout
        )
        bids_stimulation_amplitude_stepsize = widgets.BoundedFloatText(
            value=0.5,
            min=0,
            max=10,
            step=0.1,
            description='Stimulation Amplitude stepsize:',
            style=style,
            layout=layout
        )
    else:
        preset_stimL = 0
        preset_stimR = 0

    bids_stimulation_amplitude_left=widgets.BoundedFloatText(
            value=preset_stimL,
            min=0,
            max=10,
            step=0.1,
            description='Stimulation Amplitude left:',
            style=style,
            layout=layout
        )

    bids_stimulation_amplitude_right=widgets.BoundedFloatText(
            value=preset_stimR,
            min=0,
            max=10,
            step=0.1,
            description='Stimulation Amplitude right:',
            style=style,
            layout=layout
        )

    with output2:
        display(bids_reference[-1])
        if 'StimOn' in bids_acquisition[-1].value:
            for anocon in range(0, 8):
                display(bids_anodal_contact[anocon])
            if 'StimOnB' in bids_acquisition[-1].value or 'StimOnR' in bids_acquisition[-1].value:
                display(bids_stimulation_amplitude_right)
                display(bids_stimulation_frequency_right)
            if 'StimOnB' in bids_acquisition[-1].value or 'StimOnL' in bids_acquisition[-1].value:
                display(bids_stimulation_amplitude_left)
                display(bids_stimulation_frequency_left)
            for cathocon in range(0,8):
                display(bids_cathodal_contact[cathocon])


        if 'EvokRamp' in bids_task:
            display(bids_stimulation_amplitude_min)
            display(bids_stimulation_amplitude_max)
            display(bids_stimulation_amplitude_stepsize)

        display(go_to_status_description)

go_to_reference.on_click(define_reference_and_stims)
go_to_status_description = widgets.Button(
    description="define the channel status descriptions",
    style=style,
    layout=layout,
)

def define_status_description(*args):
    global bids_channel_names_widgets
    global bids_channel_names_list
    global bids_reference
    global bids_status_description_widgets
    global bids_status_description_list
    global bids_cathodal_contact
    global bids_stimulation_amplitude_left
    global bids_stimulation_frequency_left
    global bids_stimulation_amplitude_right
    global bids_stimulation_frequency_right
    global bids_anodal_contact
    bids_status_description_widgets = []
    cathocontacts = []
    for stimcon in range(0,8):
        cathocontacts.append(bids_cathodal_contact[stimcon].value)
    anocontacts = []
    for anocon in range(0, 8):
        anocontacts.append(bids_anodal_contact[anocon].value)
    for ch in bids_channel_names_list:
        if ch in cathocontacts:
            defaultvalue = 'Stimulation contact (cathode)'
        elif ch in anocontacts:
            defaultvalue = 'Stimulation contact (anode)'
        elif ch in bids_reference[-1].value:
            defaultvalue = 'Reference electrode'
        else:
            defaultvalue = 'n/a'

        bids_status_description_widgets.append(
            widgets.Combobox(
            value=defaultvalue,
            options=['Reference electrode','Stimulation contact (cathode)', 'Stimulation contact (anode)', 'Empty', 'Cable artefact'],
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
    global bids_channel_names_widgets
    global bids_channel_names_list
    global bids_reference
    global bids_status_description_widgets
    global bids_status_description_list
    global bids_cathodal_contact
    global bids_stimulation_amplitude_left
    global bids_stimulation_frequency_left
    global bids_stimulation_amplitude_right
    global bids_stimulation_frequency_right
    global bids_anodal_contact
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

    ###################### Now fill in the metadict dictionary ######################

    try:
        metadict['inputdata_location'] = bids_filechooser[-1].selected_path + os.sep + bids_filechooser[-1].selected_filename
        metadict['inputdata_fname'] = bids_filechooser[-1].selected_filename
        metadict['entities'] = {}
        metadict['entities']['subject'] = str(bids_subject_prefix.value) + str(bids_subject.value).zfill(3)
        metadict['entities']['session'] = bids_session[-1].value
        metadict['entities']['task'] = task_options[bids_task_number[-1].value][0]
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
            metadict['participants']['date_of_implantation'] = "2025-na-naT00:00:00"
        finally:
            pass
        metadict['participants']['UPDRS_III_preop_OFF'] = str()
        metadict['participants']['UPDRS_III_preop_ON'] = str()
        metadict['participants']['disease_duration'] = bids_disease_duration.value
        metadict['participants']['PD_subtype'] = bids_PD_subtype.value
        metadict['participants']['symptom_dominant_side'] = bids_symptom_dominant_side.value
        metadict['participants']['LEDD'] = 0 #bids_LEDD.value
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
        metadict['stim'] = {}
        if bids_stimulation_amplitude_left.value == 0 and ('StimOnL' in metadict['entities']['acquisition'] or 'StimOnB' in metadict['entities']['acquisition']):
            with output2:
                display('Error: stim amplitude left is 0')
                display(bids_stimulation_amplitude_left)
        if bids_stimulation_amplitude_right.value == 0 and ('StimOnR' in metadict['entities']['acquisition'] or 'StimOnB' in metadict['entities']['acquisition']):
            with output2:
                display('Error: stim amplitude right is 0')
                display(bids_stimulation_amplitude_right)
        if bids_stimulation_amplitude_left.value > 0:
            metadict['stim']['DateOfSetting'] = metadict['sessions_tsv']['acq_date']
            metadict['stim']['L'] = {}
            metadict['stim']['L']['CathodalContact'] = []
            metadict['stim']['L']['AnodalContact'] = []
            metadict['stim']['L']['StimulationAmplitude'] = bids_stimulation_amplitude_left.value
            metadict['stim']['L']['StimulationFrequency'] = bids_stimulation_frequency_left.value
            if bids_stimulation_frequency_left.value == 0 :
                with output2:
                    display('Error: stim frequency left is 0')
                    display(bids_stimulation_frequency_left)
            if 'EvokRamp' in bids_task:
                metadict['stim']['L']['StimulationAmplitudeMin'] = bids_stimulation_amplitude_min.value
                metadict['stim']['L']['StimulationAmplitudeMax'] = bids_stimulation_amplitude_max.value
                metadict['stim']['L']['StimulationAmplitudeStepsize'] = bids_stimulation_amplitude_stepsize.value
        if bids_stimulation_amplitude_right.value > 0:
            metadict['stim']['DateOfSetting'] = metadict['sessions_tsv']['acq_date']
            metadict['stim']['R'] = {}
            metadict['stim']['R']['CathodalContact'] = []
            metadict['stim']['R']['AnodalContact'] = []
            metadict['stim']['R']['StimulationAmplitude'] = bids_stimulation_amplitude_right.value
            metadict['stim']['R']['StimulationFrequency'] = bids_stimulation_frequency_right.value
            if bids_stimulation_frequency_left.value == 0:
                with output2:
                    display('Error: stim frequency right is 0')
                    display(bids_stimulation_frequency_right)
            if 'EvokRamp' in bids_task:
                metadict['stim']['R']['StimulationAmplitudeMin'] = bids_stimulation_amplitude_min.value
                metadict['stim']['R']['StimulationAmplitudeMax'] = bids_stimulation_amplitude_max.value
                metadict['stim']['R']['StimulationAmplitudeStepsize'] = bids_stimulation_amplitude_stepsize.value
        try:
            for stimcon in range(0, 8):
                #with output2:
                #    display('the cathodal contact nr')
                #    display(stimcon)
                #    display(bids_cathodal_contact[stimcon].value)
                #    display('the anodal contact nr')
                #    display(stimcon)
                #    display(bids_anodal_contact[stimcon].value)
                if '_L_' in bids_cathodal_contact[stimcon].value in bids_cathodal_contact[stimcon].value:
                    metadict['stim']['L']['CathodalContact'].append(bids_cathodal_contact[stimcon].value)
                elif '_R_' in bids_cathodal_contact[stimcon].value in bids_cathodal_contact[stimcon].value:
                    metadict['stim']['R']['CathodalContact'].append(bids_cathodal_contact[stimcon].value)
                if '_L_' in bids_anodal_contact[stimcon].value in bids_anodal_contact[stimcon].value:
                    metadict['stim']['L']['AnodalContact'].append(bids_anodal_contact[stimcon].value)
                elif '_R_' in bids_anodal_contact[stimcon].value in bids_anodal_contact[stimcon].value:
                    metadict['stim']['R']['AnodalContact'].append(bids_anodal_contact[stimcon].value)
                if bids_stimulation_amplitude_left.value > 0 and 'Ground' in bids_cathodal_contact[stimcon].value:
                    metadict['stim']['L']['CathodalContact'].append(bids_cathodal_contact[stimcon].value)
                if bids_stimulation_amplitude_right.value > 0 and 'Ground' in bids_cathodal_contact[stimcon].value:
                    metadict['stim']['R']['CathodalContact'].append(bids_cathodal_contact[stimcon].value)
                if bids_stimulation_amplitude_left.value > 0 and 'Ground' in bids_anodal_contact[stimcon].value:
                    metadict['stim']['L']['AnodalContact'].append(bids_anodal_contact[stimcon].value)
                if bids_stimulation_amplitude_right.value > 0 and 'Ground' in bids_anodal_contact[stimcon].value:
                    metadict['stim']['R']['AnodalContact'].append(bids_anodal_contact[stimcon].value)
        except:
            with output2:
                display('ERROR: bids cathodal or anodal contacts could not be assigned')
                display(bids_cathodal_contact[0:7].value)
                display(bids_anodal_contact[0:7].value)
        # some assertions

        if metadict['entities']['acquisition'] == 'StimOff':
            if bids_stimulation_amplitude_right.value > 0 or bids_stimulation_amplitude_left.value > 0:
                with output2:
                    display('Error: StimOff should have not stim amplitude')
        if 'StimOn' in metadict['entities']['acquisition']:
            if bids_stimulation_amplitude_right.value == 0 and bids_stimulation_amplitude_left.value == 0:
                with output2:
                    display('Error: StimOn should have a least one stim amplitude')

        '''
        for stimcon in range(0,8):
            if len(bids_cathodal_contact)==8 or len(bids_anodal_contact)==8:
                if (bids_cathodal_contact[stimcon].value != "") and (metadict['entities']['acquisition'] != 'StimOff') or (bids_anodal_contact[stimcon].value != "") and (metadict['entities']['acquisition'] != 'StimOff'):
                    if not 'stim' in metadict:
                        metadict['stim'] ={}
                        metadict['stim']['DateOfSetting'] = metadict['sessions_tsv']['acq_date']
                        try:
                            if bids_stimulation_amplitude_left.value > 0:
                                metadict['stim']['L'] = {}
                                metadict['stim']['L']['CathodalContact'] = []
                                metadict['stim']['L']['AnodalContact'] = []
                                metadict['stim']['L']['StimulationAmplitude'] = bids_stimulation_amplitude_left.value
                                metadict['stim']['L']['StimulationFrequency'] = bids_stimulation_frequency_left.value
                                if 'EvokRamp' in bids_task:
                                    metadict['stim']['L']['StimulationAmplitudeMin'] = bids_stimulation_amplitude_min.value
                                    metadict['stim']['L']['StimulationAmplitudeMax'] = bids_stimulation_amplitude_max.value
                                    metadict['stim']['L']['StimulationAmplitudeStepsize'] = bids_stimulation_amplitude_stepsize.value
                            if bids_stimulation_amplitude_right.value > 0:
                                metadict['stim']['R'] = {}
                                metadict['stim']['R']['CathodalContact'] = []
                                metadict['stim']['R']['AnodalContact'] = []
                                metadict['stim']['R']['StimulationAmplitude'] = bids_stimulation_amplitude_right.value
                                metadict['stim']['R']['StimulationFrequency'] = bids_stimulation_frequency_right.value
                                if 'EvokRamp' in bids_task:
                                    metadict['stim']['R']['StimulationAmplitudeMin'] = bids_stimulation_amplitude_min.value
                                    metadict['stim']['R']['StimulationAmplitudeMax'] = bids_stimulation_amplitude_max.value
                                    metadict['stim']['R']['StimulationAmplitudeStepsize'] = bids_stimulation_amplitude_stepsize.value
                        except:
                            with output2:
                                display(bids_stimulation_amplitude_left)
                                display(bids_stimulation_amplitude_right)
                                display(bids_stimulation_amplitude_left.value)
                                display(bids_stimulation_amplitude_right.value)
                                display(bids_cathodal_contact)
                    try:
                        if '_L_' or 'Ground' in bids_cathodal_contact[stimcon].value:
                            metadict['stim']['L']['CathodalContact'].append(bids_cathodal_contact[stimcon].value)
                        if '_R_' or 'Ground' in bids_cathodal_contact[stimcon].value:
                            metadict['stim']['R']['CathodalContact'].append(bids_cathodal_contact[stimcon].value)
                        if '_L_' or 'Ground' in bids_anodal_contact[stimcon].value:
                            metadict['stim']['L']['AnodalContact'].append(bids_anodal_contact[stimcon].value)
                        if '_R_' or 'Ground' in bids_anodal_contact[stimcon].value:
                            metadict['stim']['R']['AnodalContact'].append(bids_anodal_contact[stimcon].value)
                    except:
                        with output2:
                            display(bids_stimulation_amplitude_left)
                            display(bids_stimulation_amplitude_right)
                            display(bids_stimulation_amplitude_left.value)
                            display(bids_stimulation_amplitude_right.value)
                            display(bids_cathodal_contact)
            else:
                with output2:
                    print("bids_cathodal_contact")
                    print(bids_cathodal_contact)
                    print("ERROR bids_cathodal_contact shorter as 8")
                    
        bids_cathodal_contact=[]
        bids_stimulation_amplitude_left=0
        bids_stimulation_amplitude_right=0
        if 'StimOff' in metadict['entities']['acquisition']:
            metadict['stim'] = {}
        '''
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
    except:
        currentfile = bids_filechooser[-1].selected_filename
        currentfile += ".json"
        with open(currentfile, "w") as outfile:
            json.dump(metadict, outfile, indent=4)

        with output2:
            print("saving to: %.json", bids_filechooser[-1].selected_filename)
            print("ERROR information not sucessfully saved, only partially saved")
    finally:
        with output2:
            print(metadict)

save_to_json.on_click(multiplefunctions_2)


output2 = widgets.Output()

# # overwrite previous list
#     bids_channel_names_widgets = []
#     bids_channel_names_list = []
#     bids_status_description_widgets = []
#     bids_status_description_list = []


dictchannelnames = {
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
    "F6": "EMG_L_R1C1_BR_TM",
    "F2": "EMG_L_R1C2_BR_TM",
    "F1": "EMG_L_R1C3_BR_TM",
    "F5": "EMG_L_R1C4_BR_TM",
    "AF8": "EMG_L_R1C5_BR_TM",
    "AF4": "EMG_L_R1C6_BR_TM",
    "AF3": "EMG_L_R1C7_BR_TM",
    "AF7": "EMG_L_R1C8_BR_TM",
    "C2": "EMG_L_R2C1_BR_TM",
    "C6": "EMG_L_R2C2_BR_TM",
    "CP3": "EMG_L_R2C3_BR_TM",
    "C1": "EMG_L_R2C4_BR_TM",
    "C5": "EMG_L_R2C5_BR_TM",
    "FC4": "EMG_L_R2C6_BR_TM",
    "FCz": "EMG_L_R2C7_BR_TM",
    "FC3": "EMG_L_R2C8_BR_TM",
    "P5": "EMG_L_R3C1_BR_TM",
    "CP4": "EMG_L_R3C2_BR_TM",
    "CPz": "EMG_L_R3C3_BR_TM",
    "P1": "EMG_L_R3C4_BR_TM",
    "P2": "EMG_L_R3C5_BR_TM",
    "P6": "EMG_L_R3C6_BR_TM",
    "PO5": "EMG_L_R3C7_BR_TM",
    "PO3": "EMG_L_R3C8_BR_TM",
    "PO4": "EMG_L_R4C1_BR_TM",
    "PO6": "EMG_L_R4C2_BR_TM",
    "FT7": "EMG_L_R4C3_BR_TM",
    "FT8": "EMG_L_R4C4_BR_TM",
    "TP7": "EMG_L_R4C5_BR_TM",
    "TP8": "EMG_L_R4C6_BR_TM",
    "PO7": "EMG_L_R4C7_BR_TM",
    "PO8": "EMG_L_R4C8_BR_TM",
    }