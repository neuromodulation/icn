%% Initialize settings and go to root of source data
clear all, close all, clc
restoredefaultpath

addpath(fullfile('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\wjn_toolbox'));
addpath(fullfile('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\Fieldtrip_Toolbox'));
ft_defaults

% This is the output root folder for our BIDS-dataset
rawdata_root = 'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_Conversion\rawdata'
% This is the input root folder for our BIDS-dataset
sourcedata_root = 'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_Conversion\'
current_recording_folder = '531AI63_MedOff1_SelfpRotaR_StimOff_1 - 20220124T093627'
input_recording = '531AI63_MedOff1_SelfpRotaR_StimOff_1-20220124T093627.DATA.Poly5'
% Go to folder containing measurement data
cd([sourcedata_root current_recording_folder]);

%% Select input_recording, read data with Fieldtrip and inspect data with WJN Toolbox

inputfig            = [];
inputfig.dataset    = [input_recording];
inputfig.continuous = 'yes';
data = ft_preprocessing(inputfig);

wjn_plot_raw_signals(data.time{1},data.trial{1},data.label);

%% Pick only channels to keep and re-read data
% channel to keep are:
    % LFP "M" -> indicates Medtronic
    % empty LFP channel is the reference -> keep this one
    % ISO AUX -> indicates the rotameter, should be ANALOG_R_ROTA_CH
    % EEG Cx and Fz
    % ECX -> indicates ECOG
    % X Y Z (right) and X-1 Y-1 Z-1 (left) -> indicate the accelerometer
    % BIP 01 -> is EMG_R_BR_TM: you see cardiac activity in the channel
    % BIP 02 -> EMG_L_BR_TM: no activity. because it is left
    % BIP 03 : this is the ECG
% throw away
    % Counter
    % Status
    % Z-AXIS, Y-AXIS, X-AXIS
chans          = [[1:34]]; % count the channels you need to keep
outputfig            = [];
outputfig.dataset    = [input_recording];
outputfig.continuous = 'yes';
outputfig.channel    = chans;
data = ft_preprocessing(outputfig);

%% Rename channels and fix header
% To Do Jonathan: need to check the manufacturer to know the abbreviation and the number of
% channels

DBS_model = 'SenSight';
%DBS_model = 'Vercise Cartesia X';
%DBS_model = 'Other';

ECOG_model = 'TS06R-AP10X-0W6'; %this model has 6 contacts
n_ECOG_contacts = 6;

%ECOG_model ='DS12A-SP10X-000'; % this has 12 contacts
%n_ECOG_contacts = 12;

%ECOG_hemisphere = 'right';
ECOG_hemisphere = 'left';

iEEGRef = 'LFP_L_8_STN_BS'; % what is the reference?

if strcmp(DBS_model, 'SenSight')
    n_DBS_contacts = 8;
    new_chans = {
        'LFP_R_1_STN_MT'
        'LFP_R_2_STN_MT'
        'LFP_R_3_STN_MT'
        'LFP_R_4_STN_MT'
        'LFP_R_5_STN_MT'
        'LFP_R_6_STN_MT'
        'LFP_R_7_STN_MT'
        'LFP_R_8_STN_MT'
        
        'LFP_L_1_STN_MT'
        'LFP_L_2_STN_MT'
        'LFP_L_3_STN_MT'
        'LFP_L_4_STN_MT'
        'LFP_L_5_STN_MT'
        'LFP_L_6_STN_MT'
        'LFP_L_7_STN_MT'
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
        'ECG'
        'ACC_R_X_D2_TM'
        'ACC_R_Y_D2_TM'
        'ACC_R_Z_D2_TM'
        'ACC_L_X_D2_TM'
        'ACC_L_Y_D2_TM'
        'ACC_L_Z_D2_TM'
        'ANALOG_R_ROTA_CH'
        } 
elseif strcmp(DBS_model, 'Vercise Cartesia X')
    n_DBS_contacts = 16;
    new_chans = {
        'LFP_R_1_STN_BS'
        'LFP_R_2_STN_BS'
        'LFP_R_3_STN_BB'
        'LFP_R_4_STN_BS'
        'LFP_R_5_STN_BS'
        'LFP_R_6_STN_BS'
        'LFP_R_7_STN_BS'
        'LFP_R_8_STN_BS'
        'LFP_R_9_STN_BS'
        'LFP_R_10_STN_BS'
        'LFP_R_11_STN_BB'
        'LFP_R_12_STN_BS'
        'LFP_R_13_STN_BS'
        'LFP_R_14_STN_BS'
        'LFP_R_15_STN_BS'
        'LFP_R_16_STN_BS'
        'LFP_L_1_STN_BS'
        'LFP_L_2_STN_BS'
        'LFP_L_3_STN_BS'
        'LFP_L_4_STN_BS'
        'LFP_L_5_STN_BS'
        'LFP_L_6_STN_BS'
        'LFP_L_7_STN_BS'
        'LFP_L_8_STN_BS'
        'LFP_L_9_STN_BS'
        'LFP_L_10_STN_BS'
        'LFP_L_11_STN_BB'
        'LFP_L_12_STN_BS'
        'LFP_L_13_STN_BS'
        'LFP_L_14_STN_BS'
        'LFP_L_15_STN_BS'
        'LFP_L_16_STN_BS'
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
        'ECG'
        'ACC_R_X_D2_TM'
        'ACC_R_Y_D2_TM'
        'ACC_R_Z_D2_TM'
        'ACC_L_X_D2_TM'
        'ACC_L_Y_D2_TM'
        'ACC_L_Z_D2_TM'
        %'ANALOG_R_ROTA_CH'
        }
else
    error('please, specify the DBS model')
end




data.label          = new_chans;
data.hdr.nChans     = length(new_chans);
data.hdr.label      = data.label;
% Set channel types and channel units
chantype            = cell(data.hdr.nChans,1);
for ch=1: data.hdr.nChans
    if contains(new_chans{ch},'LFP')
        chantype(ch) = {'DBS'};
    elseif contains(new_chans{ch},'ECOG')
        chantype(ch) = {'ECOG'};
    elseif contains(new_chans{ch},'EEG')
        chantype(ch) = {'EEG'};
    elseif contains(new_chans{ch},'EMG')
        chantype(ch) = {'EMG'};    
    elseif contains(new_chans{ch},'ECG')
        chantype(ch) = {'ECG'};   
    else
        chantype(ch) = {'MISC'};   
    end
end

data.hdr.chantype   = chantype;
data.hdr.chanunit   = repmat({'uV'},data.hdr.nChans,1);

%% Plot data with WJN viewer to double-check
figure
wjn_plot_raw_signals(data.time{1},data.trial{1},data.label);

%% Note which channels were bad and why
%bad = {'LFP_L_7_STN_MT' 'LFP_L_8_STN_MT' 'LFP_L_9_STN_MT' 'LFP_L_16_STN_MT' 'LFP_R_7_STN_MT' 'LFP_R_8_STN_MT' 'LFP_R_9_STN_MT'};
%why = {'Stimulation contact' 'Stimulation contact' 'Stimulation contact' 'Reference electrode' 'Stimulation contact' 'Stimulation contact' 'Stimulation contact' 'Stimulation contact'};
bad ={'LFP_L_8_STN_MT'};
why = {'Reference electrode'};

bads = repmat({'good'},data.hdr.nChans,1);
bads_descr = repmat({'n/a'},data.hdr.nChans,1);
for k=1:length(new_chans)
    if ismember(new_chans{k}, bad)
        bads{k} = 'bad';
        bads_descr{k} = why{find(strcmp(bad,new_chans{k}))};
    end
end

%% Initalize containers for BIDS conversion
keySet = {'Rest', 'UPDRSIII', 'SelfpacedRotationL','SelfpacedRotationR',...
    'BlockRotationL','BlockRotationR', 'Evoked', 'SelfpacedSpeech',...
    'ReadRelaxMoveL', 'VigorStimR', 'VigorStimL', 'SelfpacedHandFlipL',...
    'SelfpacedHandFlipR', 'Free'...
    };
descrSet = {'Rest recording', ...
    'Recording performed during part III of the UPDRS (Unified Parkinson''s Disease Rating Scale) questionnaire.',...
    'Selfpaced left wrist rotations performed on custom-built analog rotameter which translates degree of rotation to volt.',...
    'Selfpaced right wrist rotations performed on custom-built analog rotameter which translates degree of rotation to volt.',...
    'Blocks of 30 seconds of rest followed by blocks of 30 seconds of continuous wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt. Performed with the left hand.',...
    'Blocks of 30 seconds of rest followed by blocks of 30 seconds of continuous wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt. Performed with the right hand.',...
    'Evoked potentials recording. Single stimulation pulses of fixed amplitude following periods of high frequency stimulation with varying amplitude (0, 1.5 and 3 mA) per block.',...
    'Selfpaced reading aloud of the fable ''The Parrot and the Cat'' by Aesop. Extended pauses in between sentences.',...
    'Block of 30 seconds of continuous left wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt followed by a block of 30 seconds of rest followed by a block of 30 seconds of reading aloud (''The Parrot and the Cat'' by Aesop). Multiple sets.',...
    'Performance of diagonal forearm movements with a cursor on a screen using a digitizing tablet. Start and stop events are visually cued on screen with a rest duration of 350 ms. 14 blocks with 32 movements each. In blocks 3-5/9-11 bilateral stimulation is applied for 300 ms if a movement is slower/faster than the previous two movements. The order of slow/fast blocks is alternated between participants.  Performed with the right hand.',...
    'Performance of diagonal forearm movements with a cursor on a screen using a digitizing tablet. Start and stop events are visually cued on screen with a rest duration of 350 ms. 14 blocks with 32 movements each. In blocks 3-5/9-11 bilateral stimulation is applied for 300 ms if a movement is slower/faster than the previous two movements. The order of slow/fast blocks is alternated between participants.  Performed with the left hand.',...
    'Selfpaced left hand rotations 180 every 10 seconds, without counting, in resting seated position.',...
    'Selfpaced right hand rotations 180 every 10 seconds, without counting, in resting seated position.',...
    'Free period, no instructions, during Dyskinesia-Protocol still recorded to monitor the increasing Dopamine-Level'...
    };
instructionSet = {'Do not move or speak and keep your eyes open.',...
    'See UPDRS questionnaire.',...
    'Perform 50 wrist rotations with your left hand with an interval of about 10 seconds. Do not count in between rotations.',...
    'Perform 50 wrist rotations with your right hand with an interval of about 10 seconds. Do not count in between rotations.',...
    'Upon the auditory command "start", perform continuous wrist rotations with your left hand, until you perceive the auditory command "stop". Perform these wrist rotations as fast as possible and with the largest possible amplitude.',...
    'Upon the auditory command "start", perform continuous wrist rotations with your right hand, until you perceive the auditory command "stop". Perform these wrist rotations as fast as possible and with the largest possible amplitude.',...
    'Do not move or speak and keep your eyes open.',...
    'Read aloud sentence by sentence the text in front of you. Leave a pause of several seconds in between sentences.',...
    'At the beginning of each block, a text will appear on the screen, specifying the task to be performed. An auditory cue will then be issued, marking the begin of your task. Perform the task until the next cue marks the end of the task. Tasks are either continuous left wrist rotation, resting with open eyes or reading aloud the text displayed on the screen.',...
    'Your task is to move your pen from one side of the screen to the other. When you see a square, please move your pen there and stay on the square until a new square appears on the other side. Then move the pen to the new square. Please move as fast as you can and touch the screen with the pen throughout the whole experiment.',...
    'Your task is to move your pen from one side of the screen to the other. When you see a square, please move your pen there and stay on the square until a new square appears on the other side. Then move the pen to the new square. Please move as fast as you can and touch the screen with the pen throughout the whole experiment.',...
    'Keep both hands resting on your legs, and rotate your left hand 180 degrees and back every 10 seconds. Do not count in between rotations.',...
    'Keep both hands resting on your legs, and rotate your right hand 180 degrees and back every 10 seconds. Do not count in between rotations.',...
    'Free period, without instructions or restrictions, of rest between Rest-measurement and Task-measurements'...
    };
task_descr = containers.Map(keySet,descrSet)
task_instr = containers.Map(keySet,instructionSet)

%% Now write data to BIDS
% data2bids function is only found in original fieldtrip toolbox
% restoredefaultpath;
% addpath(fullfile('C:\Users\richa\GitHub\fieldtrip\'));

% Initialize a 'n/a' variable for practicality
n_a = repmat({'n/a'},data.hdr.nChans,1);

% adept for each different recording
cfg = [];
cfg.method                  = 'convert';
cfg.bidsroot                = rawdata_root;
cfg.datatype                = 'ieeg';
cfg.sub                     = '009';
cfg.ses                     = 'EcogLfpMedOff01';
cfg.task                    = 'SelfpacedRotationR';
cfg.acq                     = 'StimOff01';  % add here 'Dopa00' for dyskinesia (MedOn3) recording: e.g. 'StimOff01Dopa30')
cfg.run                     = '01';

% Provide info for the scans.tsv file
% the acquisition time could be found in the folder name of the recording

cfg.scans.acq_time              = '2022-01-24T09:36:27';
if contains(cfg.ses, 'Off')
    cfg.scans.medication_state  = 'OFF';
else
    cfg.scans.medication_state  = 'ON';
end
cfg.scans.UPDRS_III             = 'n/a'; % need to be calcuated.

% Specify some general information
cfg.InstitutionName                         = 'Charite - Universitaetsmedizin Berlin, corporate member of Freie Universitaet Berlin and Humboldt-Universitaet zu Berlin, Department of Neurology with Experimental Neurology/BNIC, Movement Disorders and Neuromodulation Unit';
cfg.InstitutionAddress                      = 'Chariteplatz 1, 10117 Berlin, Germany';
cfg.dataset_description.Name                = 'BIDS_Berlin_LFP_ECOG_PD';
cfg.dataset_description.BIDSVersion         = '1.5.0';
cfg.dataset_description.License             = 'n/a';
cfg.dataset_description.Funding             = [['Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - Project-ID 424778381 - TRR 295']];
cfg.dataset_description.Authors             = {'Johannes Busch', 'Meera Chikermane', 'Katharina Faust', 'Lucia Feldmann', 'Richard Koehler', 'Andrea Kuehn', 'Roxanne Lofredi', 'Timon Merk', 'Wolf-Julian Neumann', 'Gerd-Helge Schneider', 'Ulrike Uhlig','Jeroen Habets','Jonathan Vanhoecke'};
cfg.dataset_description.Acknowledgements    = 'Special thanks to Ulrike Uhlig for their help in recording the data.';

% Provide the long description of the task and participant instructions
cfg.TaskName                = cfg.task;
cfg.TaskDescription         = task_descr(cfg.task);
cfg.Instructions            = task_instr(cfg.task);

% Provide info about recording hardware
manufacturer                        = 'TMSi';
if strcmp(manufacturer,'TMSi')
    cfg.Manufacturer                = 'Twente Medical Systems International B.V. (TMSi)';
    cfg.ManufacturersModelName      = 'Saga 64+';
    cfg.SoftwareVersions            = 'TMSi Polybench - QRA for SAGA - REV1.0.0';
    cfg.DeviceSerialNumber          = '1005190056';
    cfg.channels.low_cutoff         = repmat({'0.0'},data.hdr.nChans,1);
    cfg.channels.high_cutoff        = repmat({'2100.0'},data.hdr.nChans,1); 
    cfg.channels.high_cutoff(contains(string(new_chans),["LFP", "ECOG","EEG"])) = {'1600.0'}
%else manufacturer == 'Ripple'
%else manufacturer == 'AlphaOmega'
%else manufacturer == 'Neuronica'
else
    error('Please define the Hardware manufacturer')
end

% need to check in the LFP excel sheet on the S-drive
% Provide info about the participant	
% cfg.participants.sex                    = 'male'; -> where is this written down?
 cfg.participants.handedness             = 'right';
 cfg.participants.age                    = 59;
 cfg.participants.date_of_implantation   = '2022-01-20T00:00:00';
% cfg.participants.UPDRS_III_preop_OFF    = 'n/a';-> how to calculate?
% cfg.participants.UPDRS_III_preop_ON     = 'n/a';-> how to calculate?
 cfg.participants.disease_duration       = 7;
% cfg.participants.PD_subtype             = 'akinetic-rigid';-> where to find this?
 cfg.participants.symptom_dominant_side  = 'right';
% cfg.participants.LEDD                   = 'n/a';-> where to find this?


% TO DO:  provide a dictionary for the DBS manufacturer
% Provide info about the DBS lead

cfg.participants.DBS_target                 = 'STN';
if strcmp(DBS_model, 'SenSight')    
    cfg.participants.DBS_manufacturer           = 'Medtronic';
    cfg.participants.DBS_model                  = 'SenSight';
    cfg.participants.DBS_directional            = 'yes';
    cfg.participants.DBS_contacts               = n_DBS_contacts;
    cfg.participants.DBS_description            = '8-contact, directional DBS lead.';
elseif strcmp(DBS_model, 'Vercise Cartesia X')    
    cfg.participants.DBS_manufacturer           = 'Boston Scientific';
    cfg.participants.DBS_model                  = 'Vercise Cartesia X';
    cfg.participants.DBS_directional            = 'yes';
    cfg.participants.DBS_contacts               = n_DBS_contacts;
    cfg.participants.DBS_description            = '16-contact, directional DBS lead.';
else
    error('please, specify the DBS model')
end
    
% Provide info about the ECOG electrode
cfg.participants.ECOG_target                = 'sensorimotor cortex';
cfg.participants.ECOG_hemisphere            = ECOG_hemisphere
if strcmp(ECOG_model, 'TS06R-AP10X-0W6')
    cfg.participants.ECOG_manufacturer          = 'Ad-Tech';
    cfg.participants.ECOG_model                 = 'TS06R-AP10X-0W6';
    cfg.participants.ECOG_location              = 'subdural';
    cfg.participants.ECOG_material              = 'platinum';
    cfg.participants.ECOG_contacts              = n_ECOG_contacts;
    cfg.participants.ECOG_description           = '6-contact, 1x6 narrow-body long term monitoring strip. Platinum contacts, 10mm spacing, contact size 4.0 mm diameter/1.8 mm exposure.';
elseif strcmp(ECOG_model, 'DS12A-SP10X-000')
    cfg.participants.ECOG_manufacturer          = 'Ad-Tech';
    cfg.participants.ECOG_model                 = 'DS12A-SP10X-000';
    cfg.participants.ECOG_location              = 'subdural';
    cfg.participants.ECOG_material              = 'platinum';
    cfg.participants.ECOG_contacts              = n_ECOG_contacts;
    cfg.participants.ECOG_description           = '12-contact, 1x6 dual sided long term monitoring strip. Platinum contacts, 10mm spacing, contact size 4.0 mm diameter/2.3 mm exposure. Platinum marker.';
else
    error('Please specify the ECOG model')
end



% Provide info for the coordsystem.json file
% remains always the same in berlin
cfg.coordsystem.IntendedFor                         = "n/a"; % OPTIONAL. Path or list of path relative to the subject subfolder pointing to the structural MRI, possibly of different types if a list is specified, to be used with the MEG recording. The path(s) need(s) to use forward slashes instead of backward slashes (e.g. "ses-<label>/anat/sub-01_T1w.nii.gz").
cfg.coordsystem.iEEGCoordinateSystem                = "Other"; % REQUIRED. Defines the coordinate system for the iEEG electrodes. See Appendix VIII for a list of restricted keywords. If positions correspond to pixel indices in a 2D image (of either a volume-rendering, surface-rendering, operative photo, or operative drawing), this must be "Pixels". For more information, see the section on 2D coordinate systems
cfg.coordsystem.iEEGCoordinateUnits	                = "mm"; % REQUIRED. Units of the _electrodes.tsv, MUST be "m", "mm", "cm" or "pixels".
cfg.coordsystem.iEEGCoordinateSystemDescription	    = "MNI152 2009b NLIN asymmetric T2 template"; % RECOMMENDED. Freeform text description or link to document describing the iEEG coordinate system system in detail (e.g., "Coordinate system with the origin at anterior commissure (AC), negative y-axis going through the posterior commissure (PC), z-axis going to a mid-hemisperic point which lies superior to the AC-PC line, x-axis going to the right").
cfg.coordsystem.iEEGCoordinateProcessingDescription = "Co-registration, normalization and electrode localization done with Lead-DBS"; % RECOMMENDED. Has any post-processing (such as projection) been done on the electrode positions (e.g., "surface_projection", "none").
cfg.coordsystem.iEEGCoordinateProcessingReference	= "Horn, A., Li, N., Dembek, T. A., Kappel, A., Boulay, C., Ewert, S., et al. (2018). Lead-DBS v2: Towards a comprehensive pipeline for deep brain stimulation imaging. NeuroImage."; % RECOMMENDED. A reference to a paper that defines in more detail the method used to localize the electrodes and to post-process the electrode positions. .

% Provide columns in the electrodes.tsv

% REQUIRED. Name of the electrode
% need to build if for loop => what to do if the coordsys file already
% there is.
% here: on this place would the electrode localization file


% extract the channel names that contain LFP or ECOG
sens.label = new_chans(contains(string(new_chans),["LFP", "ECOG"]))

% Electrode positions are imported from external files (e.g. Lead-DBS
% ea_reconstruction.mat) and a sens FieldTrip struct is created 
% (see FT_DATATYPE_SENS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To Do: Richard
% if else statement for the manufacturer and per model
% this is the channel position for medtronic
sens.chanpos = [
    zeros(n_DBS_contacts, 3); ...
    zeros(n_DBS_contacts, 3); ...
    zeros(n_ECOG_contacts, 3)]; %what is this 12 refering to? I replaced it with n_ECOG_contact 
% this is for medtronic
% cfg.electrodes.size         = {
%     6 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
%     6 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
%     4.15 4.15 4.15 4.15 4.15 4.15 4.15 4.15 4.15 4.15 4.15 4.15}; %  => what are these values?
cfg.electrodes.size         = {
    6 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
    6 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
    4.15 4.15 4.15 4.15 4.15 4.15}; %  => what are these values?


% this is only for the boston scientific
% cfg.electrodes.size         = {
%     1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
%     1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
%     4.15 4.15 4.15 4.15 4.15 4.15 ...
%     }; => what are these values?
% sens.chanpos = [
%     zeros(n_DBS_contacts, 3); ...
%     zeros(n_DBS_contacts, 3); ...
%     zeros(6, 3)]; %what is this 6 refering to?
sens.elecpos = sens.chanpos;
cfg.elec     = sens;
cfg.electrodes.name = sens.label;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RECOMMENDED. Material of the electrode, e.g., Tin, Ag/AgCl, Gold
cfg.electrodes.material     = [
    repmat({'platinum/iridium'},n_DBS_contacts*2,1); 
    repmat({'platinum'}, n_ECOG_contacts ,1)];
cfg.electrodes.manufacturer = [
    repmat({cfg.participants.DBS_manufacturer},n_DBS_contacts*2,1);
    repmat({cfg.participants.ECOG_manufacturer},6,1)];
cfg.electrodes.group        = [
    repmat({'DBS_right'},n_DBS_contacts,1);
    repmat({'DBS_left'},n_DBS_contacts,1);
    repmat({'ECOG_strip'},n_ECOG_contacts,1)];
cfg.electrodes.hemisphere   = [
    repmat({'R'},n_DBS_contacts,1);
    repmat({'L'},n_DBS_contacts,1);
    repmat({upper(ECOG_hemisphere(1))},n_ECOG_contacts,1)];
% RECOMMENDED. Type of the electrode (e.g., cup, ring, clip-on, wire, needle)
cfg.electrodes.type         = [  
    repmat({'depth'},n_DBS_contacts*2,1); %=> this is DBS
    repmat({'strip'},n_ECOG_contacts,1)]; %=> this is the ECOG

% RECOMMENDED. Impedance of the electrode in kOhm
cfg.electrodes.impedance    = repmat({'n/a'},length(sens.label),1);  
cfg.electrodes.dimension    = [  
    repmat({sprintf('[1x%d]',n_DBS_contacts)},n_DBS_contacts*2,1);
    repmat({sprintf('[1x%d]',n_ECOG_contacts)},n_ECOG_contacts,1)];

% Provide special channel info
cfg.channels.name               = new_chans;
cfg.channels.type               = chantype;
cfg.channels.units              = data.hdr.chanunit;
%cfg.channels.description       = ft_getopt(cfg.channels, 'description'        , nan);  % OPTIONAL. Brief free-text description of the channel, or other information of interest. See examples below.
sf = cell(length(new_chans),1);
sf(:) = {data.fsample};
cfg.channels.sampling_frequency = sf;

%% Reference channels

cfg.ieeg.iEEGReference = iEEGRef;
typeSet = {'EEG', 'ECOG', 'DBS', 'SEEG', 'EMG', 'ECG', 'MISC'};
refSet = {iEEGRef, iEEGRef, iEEGRef, iEEGRef, 'bipolar', 'bipolar', 'n/a'};
ref_map = containers.Map(typeSet,refSet);
cfg.channels.reference = arrayfun(@(ch_type) {ref_map(ch_type{1})}, chantype);
% always notch filter on n/a
cfg.channels.notch              = n_a;

%% TO DO Jonathan
cfg.channels.group              = n_a; % => need to check with BIDS

%
cfg.channels.status             = bads;
cfg.channels.status_description = bads_descr;

% these are iEEG specific
cfg.ieeg.PowerLineFrequency     = 50;   % since recorded in the Europe
cfg.ieeg.iEEGGround             = 'Right shoulder patch';
% this is to be specified for each model
cfg.ieeg.iEEGPlacementScheme    = 'Left subdural cortical strip and bilateral subthalamic nucleus (STN) deep brain stimulation (DBS) leads.';
cfg.ieeg.iEEGElectrodeGroups    = 'ECOG_strip: 6-contact, 1x6 dual sided long term monitoring AdTech strip on left sensorimotor cortex, DBS_left: 1x16 Boston Scientific directional DBS lead (Cartesia X) in left STN, DBS_right: 1x16 Boston Scientific directional DBS lead (Cartesia X) in right STN.';
% To Do: Software filters => need to check which were used
% eg. {"Anti-aliasing filter": {"half-amplitude cutoff (Hz)": 500, "Roll-off": "6dB/Octave"}}.
cfg.ieeg.SoftwareFilters        = {'n/a'}; %MUST
% To Do: Hardware filters => need to check which were used
% eg. {"Highpass RC filter": {"Half amplitude cutoff (Hz)": 0.0159, "Roll-off": "6dB/Octave"}}.
cfg.ieeg.HardwareFilters        = {'n/a'}; %Recommended
cfg.ieeg.RecordingType          = 'continuous';
if contains(cfg.acq, 'On')
    cfg.ieeg.ElectricalStimulation  = true;
else
    cfg.ieeg.ElectricalStimulation  = false;
end
if cfg.ieeg.ElectricalStimulation
    % Enter EXPERIMENTAL stimulation settings
    % these need to be written in the lab book
    exp.DateOfSetting           = "2021-11-11";
    exp.StimulationTarget       = "STN";
    exp.StimulationMode         = "continuous";
    exp.StimulationParadigm     = "continuous stimulation";
    exp.SimulationMontage       = "monopolar";
    L.AnodalContact             = "G";
    L.CathodalContact           = "7, 8 and 9";
	L.AnodalContactDirection      = "none";
	L.CathodalContactDirection    = "omni";
	L.CathodalContactImpedance    = "n/a";
	L.StimulationAmplitude        = 2.0;
	L.StimulationPulseWidth       = 60;
	L.StimulationFrequency        = 130;
	L.InitialPulseShape           = "rectangular";
	L.InitialPulseWidth           = 60;
	L.InitialPulseAmplitude       = -1.0*L.StimulationAmplitude;
	L.InterPulseDelay             = 0;
	L.SecondPulseShape            = "rectangular";
	L.SecondPulseWidth            = 60;
	L.SecondPulseAmplitude        = L.StimulationAmplitude;
    L.PostPulseInterval           = "n/a";
    exp.Left                    = L;
    R.AnodalContact             = "G";
    R.CathodalContact           = "7, 8 and 9";
	R.AnodalContactDirection      = "none";
	R.CathodalContactDirection    = "omni";
	R.CathodalContactImpedance    = "n/a";
	R.StimulationAmplitude        = 2.0;
	R.StimulationPulseWidth       = 60;
	R.StimulationFrequency        = 130;
	R.InitialPulseShape           = "rectangular";
	R.InitialPulseWidth           = 60;
	R.InitialPulseAmplitude       = -1.0*R.StimulationAmplitude;
	R.InterPulseDelay             = 0;
	R.SecondPulseShape            = "rectangular";
	R.SecondPulseWidth            = 60;
	R.SecondPulseAmplitude        = R.StimulationAmplitude;
    R.PostPulseInterval           = "n/a";
    exp.Right                     = R;
    
    % Enter CLINICAL stimulation settings
    clin.DateOfSetting           = "2021-08-30";
    clin.StimulationTarget       = "STN";
    clin.StimulationMode         = "continuous";
    clin.StimulationParadigm     = "continuous stimulation";
    clin.SimulationMontage       = "monopolar";
    clear L R;
    L                           = "OFF";
    clin.Left                    = L;
    R.AnodalContact             = "G";
    R.CathodalContact           = "2, 3 and 4";
	R.AnodalContactDirection      = "none";
	R.CathodalContactDirection    = "omni";
	R.CathodalContactImpedance    = "n/a";
	R.StimulationAmplitude        = 1.5;
	R.StimulationPulseWidth       = 60;
	R.StimulationFrequency        = 130;
	R.InitialPulseShape           = "rectangular";
	R.InitialPulseWidth           = 60;
	R.InitialPulseAmplitude       = -1.5;
	R.InterPulseDelay             = 0;
	R.SecondPulseShape            = "rectangular";
	R.SecondPulseWidth            = 60;
	R.SecondPulseAmplitude        = 1.5;
    R.PostPulseInterval           = "n/a";
    clin.Right                    = R;
    
    param.BestClinicalSetting                = "n/a";
    param.CurrentExperimentalSetting         = exp;
    cfg.ieeg.ElectricalStimulationParameters = param;
end

% Now convert data to BIDS !
data2bids(cfg, data);