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
rawdata_root = 'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_dev\rawdata\'
intern_cfg.rawdata_root = rawdata_root;
% This is the input root folder for our BIDS-dataset

sourcedata_root = 'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_dev\rawdata2\'

% This is the folder where the JSON-file is stored
JsonFolder = pwd;
% define name of json-file generated for this session
intern_cfg.jsonfile = 'sub-001_ses-EphysMedOff01_task-BlockRotationR_acq-StimOffOn_run-01.json'; 

method = 'readjson';
[~,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg, method);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[cfg,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg);

% Now convert data to BIDS !
data2bids(cfg, intern_cfg.data);
% save configuration data
cd(JsonFolder)
% remove fields that should not be printed
intern_cfg_save = rmfield(intern_cfg,{'data','chs_other', 'rawdata_root'});
savejson('',intern_cfg_save,intern_cfg.jsonfile)


