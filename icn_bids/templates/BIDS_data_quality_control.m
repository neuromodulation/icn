% RUN FROM FOLDER CONTAINING THIS .M file and the meta data JSON-files

%% Initialize settings and go to root of source data
clear all, close all, clc  % actively clear workspace at start for better performance?
restoredefaultpath

addpath(fullfile('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\wjn_toolbox'));
addpath(fullfile('C:\Users\Jonathan\Documents\CODE\fieldtrip'));
addpath(fullfile('C:\Users\Jonathan\Documents\CODE\icn\icn_bids\templates'));

ft_defaults

cd('C:\Users\Jonathan\Documents\DATA\PROJECT_Berlin_dev')
jsonfiles = dir('*.json');

fh = figure;

for i =1:length(jsonfiles)
    %% pathing and set-up
    intern_cfg = struct();
    cfg = struct();

    % This is the output root folder for our BIDS-dataset
    rawdata_root = 'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_dev\rawdata14\';
    intern_cfg.rawdata_root = rawdata_root;
    % This is the input root folder for our BIDS-dataset

    
    %sourcedata_root = 'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_dev\rawdata\';
    sourcedata_root = 'C:\Users\Jonathan\Documents\CODE\icn\icn_bids\sub-014';
    
    % This is the folder where the JSON-file is stored
    JsonFolder = pwd;
    % define name of json-file generated for this session
    intern_cfg.jsonfile = jsonfiles(i).name; 
    %% make output dir rawdata
    if ~exist(rawdata_root,'dir'), mkdir(rawdata_root); end
   
    %% read the meta data 
    method = 'readjson';
    [cfg,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg, method);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Read data with Fieldtrip 

    inputfig            = [];
    inputfig.dataset    = [intern_cfg.inputdata_location];
    inputfig.continuous = 'yes';
    intern_cfg.data = ft_preprocessing(inputfig);

    
%     intern_cfg.data.label          = chs_final;
%     intern_cfg.data.hdr.nChans     = length(chs_final);
%     intern_cfg.data.hdr.label      = intern_cfg.data.label;
%     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Update channel naming and inspect data with WJN Toolbox
    if ~isequal(intern_cfg.data.label, intern_cfg.channels_tsv.name)
        method = 'update_channels';
        set(0, 'CurrentFigure', fh);
        clf reset;
        fh = figure('units','normalized','outerposition',[0 0 1 1]);
        wjn_plot_raw_signals(intern_cfg.data.time{1},intern_cfg.data.trial{1},intern_cfg.data.label);
        title( intern_cfg.jsonfile, 'before relabeling', 'interpreter', 'none')
        %saveas(gcf,fullfile(rawdata_root,['sub-',cfg.sub , '_ses-', cfg.ses, '_task-',cfg.task, '_acq-',cfg.acq, '_run-',num2str(cfg.run), '_BEFORE_relabeling.tif']))
        saveas(gcf,fullfile(rawdata_root,['sub-',intern_cfg.entities.subject , 'ses-', intern_cfg.entities.session, 'task-',intern_cfg.entities.task, 'acq-',intern_cfg.entities.acquisition, 'run-',num2str(intern_cfg.entities.run), '_BEFORE_relabeling.tif']))
        
        %'sub-',intern_cfg.sub , 'ses-', intern_cfg.ses, 'task-',intern_cfg.task, 'acq-',intern_cfg.acq, 'run-',intern_cfg.run, 
            
        [cfg,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg, method);
%         if isfield(intern_cfg,'poly5')
%             remove = [];
%             for i= 1:length(intern_cfg.data.label)
%                 % this maps the old channel name in the poly5 dictionary to the new channel name
%                 % note, as matlab does not allow spaces and hyphens in a
%                 % struct, the new and old names are to be found in a list
%                 % only select the first 15 characters, because that is how
%                 % fieldtrip works
%                 index = find(contains(intern_cfg.poly5.old,intern_cfg.data.label{i}(1:min(15, length(intern_cfg.data.label{i})))));
%                 intern_cfg.data.label{i} = intern_cfg.poly5.new{index};
%                 if isempty(intern_cfg.data.label{i})
%                     remove(end+1) = i;
%                 end
%             end
%             
%             intern_cfg.data.label(remove) = [];
%             intern_cfg.data.trial{1}(remove,:) = [];
%             intern_cfg.data.hdr.chanunit(remove) = [];
%             intern_cfg.data.hdr.chantype(remove) = [];   
%             intern_cfg.data.hdr.nChans     = length(intern_cfg.channels_tsv.name); %update the channel numbers
%         else
%             intern_cfg.data.label = intern_cfg.channels_tsv.name;
%         end
%         
%         intern_cfg.data.hdr.label      = intern_cfg.data.label; % update the other channel names fields
%         
        set(0, 'CurrentFigure', fh);
        clf reset;
        fh = figure('units','normalized','outerposition',[0 0 1 1]);
        wjn_plot_raw_signals(intern_cfg.data.time{1},intern_cfg.data.trial{1},intern_cfg.data.label);
        title( intern_cfg.jsonfile, 'after relabeling', 'interpreter', 'none')
        %saveas(gcf,fullfile(rawdata_root,['sub-',cfg.sub , '_ses-', cfg.ses, '_task-',cfg.task, '_acq-',cfg.acq, '_run-',num2str(cfg.run), '_AFTER_relabeling.tif']))
        saveas(gcf,fullfile(rawdata_root,['sub-',intern_cfg.entities.subject , 'ses-', intern_cfg.entities.session, 'task-',intern_cfg.entities.task, 'acq-',intern_cfg.entities.acquisition, 'run-',num2str(intern_cfg.entities.run), '_AFTER_relabeling.tif']))

        close all
    end
    
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Convert data to BIDS 
    [cfg,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg);

    % Now convert data to BIDS !
    data2bids(cfg, intern_cfg.data);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% save configuration data when needed
    % cd(JsonFolder)
    % % remove fields that should not be printed
    % intern_cfg_save = rmfield(intern_cfg,{'data','chs_other', 'rawdata_root'});
    % savejson('',intern_cfg_save,intern_cfg.jsonfile)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Quick fix for the scans.tsv file
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
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% move the config file out of the way -> inside the rawdata
    movefile( intern_cfg.jsonfile , fullfile(cfg.bidsroot) )
end