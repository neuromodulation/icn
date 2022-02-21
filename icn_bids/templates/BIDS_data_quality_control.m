% RUN FROM FOLDER CONTAINING THIS .M file and the JSON-file

%% Initialize settings and go to root of source data
clear all, close all, clc  % actively clear workspace at start for better performance?
restoredefaultpath

addpath(fullfile('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\wjn_toolbox'));
addpath(fullfile('C:\Users\Jonathan\Documents\CODE\fieldtrip'));
addpath(fullfile('C:\Users\Jonathan\Documents\CODE\icn\icn_bids\templates'));

ft_defaults
intern_cfg = struct();
cfg = struct();

% This is the output root folder for our BIDS-dataset
rawdata_root = '/Users/jeroenhabets/Desktop/TEMP/rawdata'
intern_cfg.rawdata_root = rawdata_root;
% This is the input root folder for our BIDS-dataset

sourcedata_root = 'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_Conversion\sourcedata\sub-010\ses-EcogLfpMedOn01'
current_recording_folder = '532LO56_MedOn1_SelfpRotaL_StimOff_1 - 20220204T163239';

% This is the folder where the JSON-file is stored
JsonFolder = pwd;
% define name of json-file generated for this session
intern_cfg.jsonfile = '532LO56_MedOn1_SelfpRotaL_StimOff_1-20220204T163239.DATA.Poly5.json'; 

method = 'readjson';
[~,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg, method);

%input_recording = '532LO56_MedOff1_Rest_StimOff_1-20220207T113556.DATA.Poly5'
input_recording = intern_cfg.filechooser;
% Go to folder containing measurement data
cd(fullfile(sourcedata_root, current_recording_folder));

draw_figures = true;
%% Select input_recording, read data with Fieldtrip and inspect data with WJN Toolbox

inputfig            = [];
inputfig.dataset    = [input_recording];
inputfig.continuous = 'yes';
intern_cfg.data = ft_preprocessing(inputfig);

if draw_figures
    figure('units','normalized','outerposition',[0 0 1 1])
    wjn_plot_raw_signals(intern_cfg.data.time{1},intern_cfg.data.trial{1},intern_cfg.data.label);
    cd(JsonFolder)
    saveas(gcf,[intern_cfg.jsonfile 'ORIGINAL.tif'])
    cd(fullfile(sourcedata_root, current_recording_folder));
end

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
    % BIP 04
% sub-008 has 2 x 16 DBS, 6 x ECOG, 2 EEG, 3 BIP (ECG-EMG), 6 ACC
chans          = [[1:49]]; % count the channels you need to keep eg. [[1:27],[29:33]]
outputfig            = [];
outputfig.dataset    = [input_recording];
outputfig.continuous = 'yes';
outputfig.channel    = chans;
intern_cfg.data = ft_preprocessing(outputfig);

if draw_figures
    figure
    wjn_plot_raw_signals(intern_cfg.data.time{1},intern_cfg.data.trial{1},intern_cfg.data.label);
end
%% Plot data with WJN viewer to double-check

% set folder back to JSON-folder
cd(JsonFolder);
 
% specify channels used other than DBS and ECOG
intern_cfg.chs_other = {
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
        };
method = 'update_channels';
[cfg,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg, method);
if draw_figures
    figure('units','normalized','outerposition',[0 0 1 1])
    wjn_plot_raw_signals(intern_cfg.data.time{1},intern_cfg.data.trial{1},intern_cfg.data.label);
    cd(JsonFolder)  % reset working directory again
    saveas(gcf,[intern_cfg.jsonfile 'CLEAN.tif'])
end

%% MANUAL INPUT: Define which channels were bad and why
% bad: contact names as in new figure. NOTE: Change LFP-LEAD side and Manufacturer code if necessary
% why: Common reasons: 'Reference', 'Empty', 'Stimulation contact'
% bad = {'LFP_L_7_STN_MT' 'LFP_L_8_STN_MT' 'LFP_L_9_STN_MT' 'LFP_L_16_STN_MT' 'LFP_R_7_STN_MT' 'LFP_R_8_STN_MT' 'LFP_R_9_STN_MT'};
% why = {'Stimulation contact' 'Stimulation contact' 'Stimulation contact' 'Reference electrode' 'Stimulation contact' 'Stimulation contact' 'Stimulation contact' 'Stimulation contact'};
intern_cfg.bad ={'LFP_L_1_STN_MT','ECOG_R_1_SMC_AT'}%,'LFP_R_5_STN_MT'};%, 'LFP_L_4_STN_MT','LFP_L_3_STN_MT','LFP_L_2_STN_MT','LFP_R_4_STN_MT','LFP_R_3_STN_MT','LFP_R_2_STN_MT'};
intern_cfg.why = {'Reference electrode','empty'}%,'empty'}%,'Stimulation contact', 'Stimulation contact','Stimulation contact', 'Stimulation contact','Stimulation contact', 'Stimulation contact'};
intern_cfg.iEEGRef ='LFP_L_1_STN_MT';  % define IEEG-reference here again


%% MANUAL INPUT: Electrode location coordinates and Stimulation-settings; written into JSON when overwrite is True
overwrite = false;  % if electrode coordinates are added for one session, it is not required to overwrite for every recording
if overwrite

%     intern_cfg.ECOG_localization =[
%      -39, -35.5, 73;
%      -38.5, -24.5, 71;
%      -38, -15, 68;
%      -36.5, -6.5, 65.5;
%      -34, 5, 62.5;
%     -33.5, 16, 59;
%         ];

    %ECOG 1
    %to
    %ECOG 6 in MNI coords
    intern_cfg.stim = false; %was there stimulation?
    if intern_cfg.stim
        intern_cfg.stim = struct();
        intern_cfg.stim.DateOfSetting = '2022-02-04';
        intern_cfg.stim.L.CathodalContact = {'LFP_L_2_STN_MT','LFP_L_3_STN_MT','LFP_L_4_STN_MT'};
        intern_cfg.stim.L.StimulationAmplitude = 2.5;
        intern_cfg.stim.L.StimulationFrequency = 130;
        intern_cfg.stim.R.CathodalContact ={'LFP_R_2_STN_MT','LFP_R_3_STN_MT','LFP_R_4_STN_MT'};
        intern_cfg.stim.R.StimulationAmplitude = 2.5;
        intern_cfg.stim.R.StimulationFrequency = 130;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[cfg,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg);

% Now convert data to BIDS !
data2bids(cfg, intern_cfg.data);
% save configuration data
cd(JsonFolder)
% remove fields that should not be printed
intern_cfg_save = rmfield(intern_cfg,{'data','chs_other', 'rawdata_root'});
% if isfield(intern_cfg_save.stim, 'DateOfSetting')
%     intern_cfg_save.DateOfSetting = intern_cfg_save.stim.DateOfSetting;
%     intern_cfg_save.stim = rmfield(intern_cfg_save.stim, 'DateOfSetting');
% end
savejson('',intern_cfg_save,intern_cfg.jsonfile)


