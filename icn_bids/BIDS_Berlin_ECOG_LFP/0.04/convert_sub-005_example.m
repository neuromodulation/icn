%% Settings
clear all, close all, clc
restoredefaultpath

addpath(fullfile('C:\Users\richa\GitHub\fieldtrip\'));
ft_defaults
addpath(fullfile('C:\Users\richa\GitHub\wjn_toolbox\'));
root = 'C:\Users\richa\OneDrive - Charité - Universitätsmedizin Berlin\LFP-Labor\Datasets\BIDS_Berlin_LFP_ECOG_PD\sourcedata\sub-511SA77\ses-20210414\'
outpath = 'C:\Users\richa\OneDrive - Charité - Universitätsmedizin Berlin\LFP-Labor\Datasets\BIDS_Berlin_LFP_ECOG_PD\derivatives\sub-511SA77\matlabfiles\ses-20210414\'
cd(root)

%% Go to select folder
cd('005_MedOff02_ReadRelaxMovL_StimOff_2 - 19000106T111657');

%% Select file and get headers
file = '005_MedOff02_ReadRelaxMovL_StimOff_2-19000106T111657.DATA.Poly5';
new_filename = 'sub-005_ses-EphysMedOff02_task-ReadRelaxMoveL_acq-StimOff_run-01_ieeg';
hdr = ft_read_header([file]);

%% Read in data to fieldtrip
cfg            = [];
cfg.dataset    = [file];
cfg.continuous = 'yes';
data2 = ft_preprocessing(cfg);

wjn_plot_raw_signals(data2.time{1},data2.trial{1},data2.label);

%% Pick channels and re-read data
%data2 = ft_read_data([file 'nf6'], 'header', hdr_nf6, 'chanindx', chans)
chans          = [1:25];
cfg            = [];
cfg.dataset    = [file];
cfg.continuous = 'yes';
%cfg.channel    = {'all', '-hifreq*'};
cfg.channel    = chans;
data2 = ft_preprocessing(cfg);

%wjn_plot_raw_signals(data2.time{1},data2.trial{1},data2.label);

%% Rename channels and fix header
new_chans = {
        'LFP_R_1_STN_MT'
        'LFP_R_234_STN_M'
        'LFP_R_567_STN_M'
        'LFP_R_8_STN_MT'
        'LFP_L_1_STN_MT'
        'LFP_L_234_STN_M'
        'LFP_L_567_STN_M'
        'LFP_L_8_STN_MT'
        'ECOG_L_1_SMC_AT'
        'ECOG_L_2_SMC_AT'
        'ECOG_L_3_SMC_AT'
        'ECOG_L_4_SMC_AT'
        'ECOG_L_5_SMC_AT'
        'ECOG_L_6_SMC_AT'
        'EEG_Cz_TM'
        'EEG_Fz_TM'
        'EMG_R_BR_TM'
        'EMG_L_BR_TM'
        'ACC_R_X_D2_TM'
        'ACC_R_Y_D2_TM'
        'ACC_R_Z_D2_TM'
        'ACC_L_X_D2_TM'
        'ACC_L_Y_D2_TM'
        'ACC_L_Z_D2_TM'
        'ANALOG_L_ROTA_C'
        };
for k=1:length(new_chans)
    if length(new_chans{k}) > 15
        disp(['Cropping the following channel to 15 chars: ' new_chans{k}])
        new_chans{k}=new_chans{k}(1:15);
    end
end
data2.label = new_chans;
data2.hdr.nChans = length(new_chans);
data2.hdr.label = data2.label;
chantype = cell(length(new_chans),1);
for k=1:length(new_chans)
    if contains(new_chans{k}, 'LFP')
        chantype(k) = {'SEEG'};
    elseif contains(new_chans{k}, 'ECOG')
        chantype(k) = {'ECOG'};
    elseif contains(new_chans{k}, 'EEG')
        chantype(k) = {'EEG'};
    elseif contains(new_chans{k}, 'EMG')
        chantype(k) = {'EMG'}; 
    else
        chantype(k) = {'MISC'};
    end
end
data2.hdr.chantype = chantype;
chanunit = cell(length(new_chans),1);
chanunit(:) = {'uV'};
data2.hdr.chanunit = chanunit;
wjn_plot_raw_signals(data2.time{1},data2.trial{1},data2.label)

%% Note which channels were bad and why
bad = {'LFP_L_1_STN_MT', 'LFP_L_567_STN_M'};
why = {'Reference channel', 'Cable broken'};
bads = cell(length(new_chans),1);
bads(:) = {'good'};
bads_descr = cell(length(new_chans),1);
bads_descr(:) = {'n/a'};
for k=1:length(new_chans)
    if ismember(new_chans{k}, bad)
        bads{k} = 'bad'
        bads_descr{k} = why{find(strcmp(bad,new_chans{k}))}
    end
end

%% Write out raw data to FieldTrip
ft_write_data([outpath new_filename '_ft'], data2, 'header', data2.hdr, 'chanindx', chans, 'dataformat', 'matlab');
test = load([outpath new_filename '_ft'])

%% Now convert and write to spm
restoredefaultpath;
addpath(fullfile('C:\Users\richa\GitHub\spm12\'));
spm('defaults','eeg');
D = spm_eeg_ft2spm(data2, [outpath new_filename '_spm']);
cd(outpath);

%% Initalize containers for BIDS conversion
keySet = {'Rest', 'UPDRSIII', 'SelfpacedRotationL','SelfpacedRotationR',...
    'BlockRotationL','BlockRotationR', 'Evoked', 'SelfpacedSpeech',...
    'ReadRelaxMoveL', 'Transition'};
descrSet = {'Rest recording', ...
    'Recording performed during part III of the UPDRS (Unified Parkinson''s Disease Rating Scale) questionnaire.',...
    'Selfpaced left wrist rotations performed on custom-built analog rotameter which translates degree of rotation to volt.',...
    'Selfpaced right wrist rotations performed on custom-built analog rotameter which translates degree of rotation to volt.',...
    'Blocks of 30 seconds of rest followed by blocks of 30 seconds of continuous wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt. Performed with the left hand.',...
    'Blocks of 30 seconds of rest followed by blocks of 30 seconds of continuous wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt. Performed with the right hand.',...
    'Evoked potentials recording. Single stimulation pulses of fixed amplitude following periods of high frequency stimulation with varying amplitude (0, 1.5 and 3 mA) per block.',...
    'Selfpaced reading aloud of the fable ''The Parrot and the Cat'' by Aesop. Extended pauses in between sentences.',...
    'Block of 30 seconds of continuous left wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt followed by a block of 30 seconds of rest followed by a block of 30 seconds of reading aloud (''The Parrot and the Cat'' by Aesop). Multiple sets.',...
    'Recording during transition from dopaminergic medication OFF to medication ON state.'};
instructionSet = {'Do not move or speak and keep your eyes open.',...
    'See UPDRS questionnaire.',...
    'Perform 50 wrist rotations with your left hand with an interval of about 10 seconds. Do not count in between rotations.',...
    'Perform 50 wrist rotations with your right hand with an interval of about 10 seconds. Do not count in between rotations.',...
    'Upon the auditory command "start", perform continuous wrist rotations with your left hand, until you perceive the auditory command "stop". Perform these wrist rotations as fast as possible and with the largest possible amplitude.',...
    'Upon the auditory command "start", perform continuous wrist rotations with your right hand, until you perceive the auditory command "stop". Perform these wrist rotations as fast as possible and with the largest possible amplitude.',...
    'Do not move or speak and keep your eyes open.',...
    'Read aloud sentence by sentence the text in front of you. Leave a pause of several seconds in between sentences.',...
    'At the beginning of each block, a text will appear on the screen, specifying the task to be performed. An auditory cue will then be issued, marking the begin of your task. Perform the task until the next cue marks the end of the task. Tasks are either continuous left wrist rotation, resting with open eyes or reading aloud the text displayed on the screen.',...
    'No specific instructions were given.'}
task_descr = containers.Map(keySet,descrSet)
task_instr = containers.Map(keySet,instructionSet)

%% Now write data to BIDS
% data2bids function is only found in original fieldtrip toolbox
restoredefaultpath;
addpath(fullfile('C:\Users\richa\GitHub\fieldtrip\'));

n_a = cell(length(new_chans),1);
n_a(:) = {'n/a'};

cfg = [];
cfg.method                  = 'convert';
cfg.bidsroot                = 'C:/Users/richa/OneDrive - Charité - Universitätsmedizin Berlin/LFP-Labor/Datasets/BIDS_Berlin_LFP_ECOG_PD/rawdata/';
cfg.datatype                = 'ieeg';
cfg.sub                     = '005'; %anonymized
cfg.ses                     = 'EphysMedOff02';
cfg.task                    = 'ReadRelaxMoveL';
cfg.acq                     = 'StimOff';
cfg.run                     = '01';

% provide info for the scans.tsv file
cfg.scans.medication_state   = 'OFF';
cfg.scans.acq_time          = '1900-01-06T11:16:57'; %anonymized

% provide the long description of the task and participant instructions
cfg.TaskName                = cfg.task;
cfg.TaskDescription         = task_descr(cfg.task);
cfg.Instructions            = task_instr(cfg.task);

% Info about recording hardware
manufacturer                        = 'TMSi';
if manufacturer == 'TMSi'
    cfg.Manufacturer                = 'Twente Medical Systems International B.V. (TMSi)';
    cfg.ManufacturersModelName      = 'Saga 64+';
    cfg.SoftwareVersions            = 'TMSi Polybench - QRA for SAGA - REV1.0.0';
    cfg.DeviceSerialNumber          = '1005190056';
else
    disp('No valid manufacturer found.')
    cfg.Manufacturer                = 'n/a';
    cfg.ManufacturersModelName      = 'n/a';
    cfg.SoftwareVersions            = 'n/a';
    cfg.DeviceSerialNumber          = 'n/a';
end

% info about the participant	
cfg.participants.sex                    = 'female';
cfg.participants.handedness             = 'right';
cfg.participants.age                    = 43;
cfg.participants.date_of_implantation   = '1900-01-01T00:00:00'; %anonymized
cfg.participants.UPDRS_III_preop_OFF    = 43;
cfg.participants.UPDRS_III_preop_ON     = 11;
cfg.participants.disease_duration       = 4;
cfg.participants.PD_subtype             = 'akinetic-rigid';
cfg.participants.symptom_dominant_side  = 'left';
cfg.participants.LEDD                   = 1221;
% info about the DBS electrode
cfg.participants.DBS_target                 = 'STN';
cfg.participants.DBS_manufacturer           = 'Medtronic';
cfg.participants.DBS_model                  = 'SenSight';
cfg.participants.DBS_directional            = 'yes';
cfg.participants.DBS_contacts               = 8;
cfg.participants.DBS_description            = '8-contact, directional DBS lead.';

% info about the ECOG electrode
cfg.participants.ECOG_target                = 'sensorimotor cortex';
cfg.participants.ECOG_hemisphere            = 'right';
cfg.participants.ECOG_manufacturer          = 'Ad-Tech';
cfg.participants.ECOG_model                 = 'TS06R-AP10X-0W6';
cfg.participants.ECOG_location              = 'subdural';
cfg.participants.ECOG_material              = 'platinum';
cfg.participants.ECOG_contacts              = 6;
cfg.participants.ECOG_description           = '6-contact, 1x6 narrow-body LTM strip. Platinum contacts, 10mm spacing';

% specify some general information that will be added to the eeg.json file
cfg.InstitutionName                         = 'Charite - Universitaetsmedizin Berlin, corporate member of Freie Universitaet Berlin and Humboldt-Universitaet zu Berlin, Department of Neurology with Experimental Neurology/BNIC, Movement Disorders and Neuromodulation Unit';
cfg.InstitutionAddress                       = 'Chariteplatz 1, 10117 Berlin, Germany';
cfg.dataset_description.Name                = 'BIDS_Berlin_LFP_ECOG_PD';
cfg.dataset_description.BIDSVersion         = '1.5.0';
cfg.dataset_description.License             = 'n/a';
cfg.dataset_description.Funding             = [['Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - Project-ID 424778381 - TRR 295']];
cfg.dataset_description.Authors             = {'Johannes Busch', 'Lucia Feldmann', 'Richard Koehler', 'Andrea Kuehn', 'Roxanne Lofredi', 'Timon Merk', 'Wolf-Julian Neumann', 'Gerd-Helge Schneider', 'Ulrike Uhlig'};

% channel info
cfg.channels.name               = new_chans;
cfg.channels.type               = chantype;
cfg.channels.units              = chanunit;
%cfg.channels.description        = ft_getopt(cfg.channels, 'description'        , nan);  % OPTIONAL. Brief free-text description of the channel, or other information of interest. See examples below.
sf = cell(length(new_chans),1);
sf(:) = {data2.fsample};
cfg.channels.low_cutoff         = n_a;
cfg.channels.high_cutoff        = n_a;
cfg.channels.reference          = n_a;
cfg.channels.group              = n_a;
cfg.channels.sampling_frequency = sf;
cfg.channels.notch              = n_a;
cfg.channels.status             = bads;
cfg.channels.status_description = bads_descr;

% these are iEEG specific
cfg.ieeg.PowerLineFrequency     = 50;   % since recorded in the Europe
cfg.ieeg.iEEGReference          = 'LFP_L_1_STN_MT'; % as stated in info
cfg.ieeg.iEEGGround             = 'Right shoulder patch';
cfg.ieeg.iEEGPlacementScheme    = 'Right subdural cortical strip and bilateral subthalamic nucleus (STN) deep brain stimulation (DBS) leads.';
cfg.ieeg.iEEGElectrodeGroups    = 'ECOG_strip: 1x6 AdTech strip on right sensorimotor cortex, DBS_left: 1x8 Medtronic directional DBS lead (SenSight) in left STN, DBS_right: 1x8 Medtronic directional DBS lead (SenSight) in right STN.';
cfg.ieeg.SoftwareFilters        = 'n/a';
cfg.ieeg.HardwareFilters        = 'n/a';
cfg.ieeg.RecordingType          = 'continuous';
if contains(cfg.acq,'On')
    cfg.ieeg.ElectricalStimulation  = true;
    A1 = 'monopolar'; % Left Stimulation mode
    A2 = 'LFP_L_2_STN_MT, LFP_L_3_STN_MT, LFP_L_4_STN_MT'; % Left Stimulation contacts
    A3 = '2.0 mA'; % Left Amplitude
    A4 = '130 Hz'; % Left Frequency
    A5 = '60 us'; % Left pulse width
    A6 = 'none'; % Left Interpulse delay
    A7 = 'negative'; % Left Initial pulse
    A8 = 'monopolar'; % Left Stimulation mode
    A9 = 'LFP_R_5_STN_MT, LFP_R_6_STN_MT, LFP_R_7_STN_MT'; % Right Stimulation contacts
    A10 = '3.0 mA'; % Right Amplitude
    A11 = '130 Hz'; % Right Frequency
    A12 = '60 us'; % Right pulse width
    A13 = 'none'; % Right Interpulse delay
    A14 = 'negative'; % Right Initial pulse direction (negative or positive)
    stimparams = ['"Left mode": %s, "Left contacts": %s, '...
        '"Left amplitude": %s, "Left frequency": %s, '...
        '"Left pulse width": %s, "Left interpulse delay": %s, "Left initial pulse": %s, '...
        '"Right mode": %s, "Right contacts": %s, '...
        '"Right amplitude": %s, "Right frequency": %s, '...
        '"Right pulse width": %s, "Right interpulse delay": %s, "Left initial pulse": %s'];
    cfg.ieeg.ElectricalStimulationParameters = sprintf(stimparams,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12);
else
    cfg.ieeg.ElectricalStimulation  = false;
end

data2bids(cfg, data2);
