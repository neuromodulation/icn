# conda install -c conda-forge ipywidgets
# pip install ipywidgets
import ipywidgets as widgets
from IPython.display import display

# pip install ipyfilechooser
from ipyfilechooser import FileChooser
import os

import json

style = {"description_width": "300px"}
layout = {"width": "800px"}

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
    options=["n/a", "SMC"],
    description="ECOG target",
    style=style,
    layout=layout,
)
bids_ECOG_hemisphere = widgets.RadioButtons(
    options=[
        "n/a",
        "R",
        "L",
    ],
    description="ECOG hemisphere",
    style=style,
    layout=layout,
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
    ("VigorStimR", 10),
    ("VigorStimL", 11),
    ("SelfpacedHandTapL", 12),
    ("SelfpacedHandTapR", 13),
    ("Free", 14),
]


def go_to_subsession(*args):
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
        "Performance of diagonal forearm movements with a cursor on a screen using a digitizing tablet. Start and stop events are visually cued on screen with a rest duration of 350 ms. 14 blocks with 32 movements each. In blocks 3-5/9-11 bilateral stimulation is applied for 300 ms if a movement is slower/faster than the previous two movements. The order of slow/fast blocks is alternated between participants.  Performed with the right hand.",
        "Performance of diagonal forearm movements with a cursor on a screen using a digitizing tablet. Start and stop events are visually cued on screen with a rest duration of 350 ms. 14 blocks with 32 movements each. In blocks 3-5/9-11 bilateral stimulation is applied for 300 ms if a movement is slower/faster than the previous two movements. The order of slow/fast blocks is alternated between participants.  Performed with the left hand.",
        "Selfpaced left hand tapping, circa every 10 seconds, without counting, in resting seated position."
        "Selfpaced right hand tapping, circa every 10 seconds, without counting, in resting seated position."
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
        "Your task is to move your pen from one side of the screen to the other. When you see a square, please move your pen there and stay on the square until a new square appears on the other side. Then move the pen to the new square. Please move as fast as you can and touch the screen with the pen throughout the whole experiment.",
        "Your task is to move your pen from one side of the screen to the other. When you see a square, please move your pen there and stay on the square until a new square appears on the other side. Then move the pen to the new square. Please move as fast as you can and touch the screen with the pen throughout the whole experiment.",
        "Keep both hands resting on your legs, and tap with your left hand by raising the hand and fingers of your left hand, without letting the arm be lifted from the leg. Do not count in between rotations."
        "Keep both hands resting on your legs, and tap with your right hand by raising the hand and fingers of your right hand, without letting the arm be lifted from the leg. Do not count in between rotations."
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
            save_to_json,
        )


specify_file.on_click(go_to_subsession)
save_to_json = widgets.Button(
    description="Save this meta data to json",
    style=style,
    layout=layout,
)


def multiplefunctions(*args):
    save_all_information(*args)
    go_to_subsession(*args)


def save_all_information(*args):
    # All the vars that I want to get start with bids_
    bidsdict = {}
    currentfile = []
    prefix = "bids_"
    sourcedict = globals().copy()

    for v in sourcedict:

        if v.startswith(prefix):

            if v == "bids_filechooser":
                with output2:

                    print("saving to: %.json", sourcedict[v][-1].selected_filename)
                val = sourcedict[v][-1].selected_filename
                currentfile = val
            elif v == "bids_date_of_implantation":
                try:
                    val = bids_date_of_implantation.value
                    val = val.strftime("%m-%d-%YT00:00:00")
                except:
                    val = "unknown"
                finally:
                    pass
            elif v == "bids_task":
                val = task_options[bids_task[-1].value][0]
            elif v == "bids_subject":
                val = str(bids_subject.value).zfill(3)

            elif type(sourcedict[v]) == list:
                val = sourcedict[v][-1].value
            else:
                val = sourcedict[v].value

            bidsdict[v[len(prefix) :]] = val
    if not currentfile:
        with output2:
            print(currentfile)
            print("The information could not be saved, please select file below")

    else:
        currentfile += ".json"
        with open(currentfile, "w") as outfile:
            json.dump(bidsdict, outfile, indent=4)

        with output2:
            print("information is saved and cannot be changed")


save_to_json.on_click(multiplefunctions)


output2 = widgets.Output()
