% RUN FROM FOLDER CONTAINING THIS .M file and the JSON-file

%% Initialize settings and go to root of source data
clear all, close all, clc  % actively clear workspace at start for better performance?
restoredefaultpath

addpath(fullfile('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\wjn_toolbox'));
addpath(fullfile('C:\Users\Jonathan\Documents\CODE\fieldtrip'));
addpath(fullfile('C:\Users\Jonathan\Documents\CODE\icn\icn_bids\templates'));

ft_defaults
jsonfiles = dir('*.json');
for i =1:length(jsonfiles)
    intern_cfg = struct();
    cfg = struct();

    % This is the output root folder for our BIDS-dataset
    rawdata_root = 'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_dev\rawdata2\';
    intern_cfg.rawdata_root = rawdata_root;
    % This is the input root folder for our BIDS-dataset

    sourcedata_root = 'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_dev\rawdata\';

    % This is the folder where the JSON-file is stored
    JsonFolder = pwd;
    % define name of json-file generated for this session
    intern_cfg.jsonfile = jsonfiles(i).name; 


    %%
    method = 'readjson';
    [~,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg, method);
    %% Select input_recording, read data with Fieldtrip and inspect data with WJN Toolbox

    inputfig            = [];
    inputfig.dataset    = [intern_cfg.inputdata_location];
    inputfig.continuous = 'yes';
    intern_cfg.data = ft_preprocessing(inputfig);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [cfg,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg);

    % Now convert data to BIDS !
    data2bids(cfg, intern_cfg.data);
    % save configuration data
    % cd(JsonFolder)
    % % remove fields that should not be printed
    % intern_cfg_save = rmfield(intern_cfg,{'data','chs_other', 'rawdata_root'});
    % savejson('',intern_cfg_save,intern_cfg.jsonfile)


    scans_json_fname = sprintf('sub-%s_ses-%s_scans.json',cfg.sub,cfg.ses);

    scans_json.acq_time.Description         = 'date of acquistion in the format YYYY-MM-DDThh:mm:ss';
    scans_json.acq_time.Units               = 'datetime';
    scans_json.acq_time.TermURL             = char("https:\\tools.ietf.org\html\rfc3339#section-5.6");
    scans_json.medication_sate.Description  = 'state of medication during recording';
    scans_json.medication_sate.Levels.OFF   = 'OFF parkinsonian medication';
    scans_json.medication_sate.Levels.ON    = 'ON parkinsonian medication';
    scans_json.UPDRS_III.Description        = char("Score of the unified Parkinson's disease rating scale (UPDRS) part III, as determined on the day of recording.");
    scans_json.UPDRS_III.TermURL            = char("https:\\doi.org\10.1002\mds.10473");

    fileID = fopen(fullfile(cfg.bidsroot,cfg.sub,cfg.ses,scans_json_fname));
    savejson('',scans_json,scans_json_fname)
    movefile(scans_json_fname,fullfile(cfg.bidsroot,['sub-' cfg.sub],['ses-' cfg.ses]))
    
    %move the config file out of the way
    movefile( intern_cfg.jsonfile , fullfile(cfg.bidsroot) )
end

