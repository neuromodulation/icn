%% Initialize settings and go to root of source data
clear all, close all, clc
restoredefaultpath

addpath(fullfile('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\wjn_toolbox'));
addpath(fullfile('C:\Users\Jonathan\Documents\CODE\fieldtrip'));
addpath(fullfile('C:\Users\Jonathan\Documents\VSCODE'));
ft_defaults
intern_cfg = struct();
cfg = struct();

% This is the output root folder for our BIDS-dataset
rawdata_root = 'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_Conversion\rawdata3'
intern_cfg.rawdata_root = rawdata_root;
% This is the input root folder for our BIDS-dataset
sourcedata_root = 'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_Conversion\sourcedata\sub-009\ses-EcogLfpMedOn02'
current_recording_folder = '531AI63_MedOn2_FreeDopa25_StimOff_1 - 20220125T105610';

JsonFolder = pwd;
intern_cfg.jsonfile = '531AI63_MedOn2_FreeDopa25_StimOff_1-20220125T105610.DATA.Poly5.json'; 
method = 'readjson';
[~,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg, method);

%input_recording = '531AI63_MedOff1_SelfpRotaR_StimOff_1-20220124T093627.DATA.Poly5'
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
chans          = [[1:33]]; % count the channels you need to keep
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
    cd(JsonFolder)
    saveas(gcf,[intern_cfg.jsonfile 'CLEAN.tif'])
end

%% Note which channels were bad and why
%bad = {'LFP_L_7_STN_MT' 'LFP_L_8_STN_MT' 'LFP_L_9_STN_MT' 'LFP_L_16_STN_MT' 'LFP_R_7_STN_MT' 'LFP_R_8_STN_MT' 'LFP_R_9_STN_MT'};
%why = {'Stimulation contact' 'Stimulation contact' 'Stimulation contact' 'Reference electrode' 'Stimulation contact' 'Stimulation contact' 'Stimulation contact' 'Stimulation contact'};
intern_cfg.bad ={'LFP_R_1_STN_MT', 'LFP_R_5_STN_MT'};
intern_cfg.why = {'Reference electrode','empty'};
intern_cfg.iEEGRef ='LFP_R_1_STN_MT';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[cfg,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg);

% Now convert data to BIDS !
data2bids(cfg, intern_cfg.data);
% save configuration data
cd(JsonFolder)
intern_cfg = rmfield(intern_cfg,{'data','chs_other', 'rawdata_root'});
jsonencode(intern_cfg)
savejson('',intern_cfg,intern_cfg.jsonfile)