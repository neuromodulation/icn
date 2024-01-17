% RUN FROM FOLDER CONTAINING THIS .M file and the meta data JSON-files

%% Initialize settings and go to root of source data
clear all, close all, clc  % actively clear workspace at start for better performance?
restoredefaultpath

addpath(fullfile('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\wjn_toolbox'));
addpath(fullfile('C:\Users\Jonathan\Documents\CODE\fieldtrip'));
addpath(fullfile('C:\Users\Jonathan\Documents\CODE\icn\icn_bids\templates'));

ft_defaults

fg = figure(1);
%% set up pathing
% this is where the meta json files are located
addpath(fullfile('C:\Users\Jonathan\Documents\DATA\PROJECT_Berlin_dev\'));
cd('C:\Users\Jonathan\Documents\DATA\PROJECT_Berlin_dev\')
% This is the output root folder for our BIDS-dataset
rawdata_root = 'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_dev\rawdata_update12\';
 % This is the input root folder for our BIDS-dataset
% sourcedata_root = 'C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_dev\rawdata10c\';
sourcedata_root = 'C:\Users\Jonathan\Documents\CODE\icn\icn_bids\';
%% set up conversion intensions
use_dummy_data = false; %for updating metadata files
% hard_coded_channel_renaming=false;
% hard_coded_reference=false;
%% let's start
jsonfiles = dir('*.json');
%% make output dir rawdata
if ~exist(rawdata_root,'dir'), mkdir(rawdata_root); end
for i =1:length(jsonfiles)
    %% set-up
    intern_cfg = struct();
    cfg = struct();
    intern_cfg.rawdata_root = rawdata_root;
   
    % This is the folder where the JSON-file is stored
    JsonFolder = pwd;
    % define name of json-file generated for this session
    intern_cfg.jsonfile = jsonfiles(i).name; 
   
   
    %% read the meta data 
    method = 'readjson';
    [cfg,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg, method);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Read data with Fieldtrip 
    inputfig = [];
    inputfig.dataset    = [intern_cfg.inputdata_location]; %not correct json file if error
    inputfig.continuous = 'yes';
    if use_dummy_data
        inputfig.trl = [1,2,0];
    end
    intern_cfg.data = ft_preprocessing(inputfig);

    
%     intern_cfg.data.label          = chs_final;
%     intern_cfg.data.hdr.nChans     = length(chs_final);
%     intern_cfg.data.hdr.label      = intern_cfg.data.label;
%     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Update channel naming and inspect data with WJN Toolbox
    % update from intern_cfg.channels_tsv.name to the intern_cfg.data.label
%     if hard_coded_channel_renaming
%         idx = startsWith(intern_cfg.channels_tsv.name,'X');
%         intern_cfg.channels_tsv.name(idx) = {'ACC_R_X_D2_TM'};
%         idx = startsWith(intern_cfg.channels_tsv.name,'Y');
%         intern_cfg.channels_tsv.name(idx) = {'ACC_R_Y_D2_TM'};
%         idx = startsWith(intern_cfg.channels_tsv.name,'Z');
%         intern_cfg.channels_tsv.name(idx) = {'ACC_R_Z_D2_TM'};
%     end
%     if hard_coded_reference
%         intern_cfg.ieeg.iEEGReference = 'LFP_L_01_STN_MT';
%     end
    
    
    if ~isequal(intern_cfg.data.label, intern_cfg.channels_tsv.name)
        if use_dummy_data
            error('no updating from channels')
        end
        
        method = 'update_channels';
        clf('reset')
        set(0,'CurrentFigure',fg);
        set(fg,'units','normalized','outerposition',[0 0 1 1]);
        wjn_plot_raw_signals(intern_cfg.data.time{1},intern_cfg.data.trial{1},intern_cfg.data.label);
        title( intern_cfg.jsonfile, 'before relabeling', 'interpreter', 'none')
        saveas(gcf,fullfile(rawdata_root,['sub-',intern_cfg.entities.subject , '_ses-', intern_cfg.entities.session, '_task-',intern_cfg.entities.task, '_acq-',intern_cfg.entities.acquisition, '_run-',num2str(intern_cfg.entities.run), '_BEFORE_relabeling.png']))
        
        [cfg,intern_cfg] =BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg, method);
      
        clf('reset')
        set(0,'CurrentFigure',fg);
        set(fg,'units','normalized','outerposition',[0 0 1 1]);
        wjn_plot_raw_signals(intern_cfg.data.time{1},intern_cfg.data.trial{1},intern_cfg.data.label);
        title( intern_cfg.jsonfile, 'after relabeling', 'interpreter', 'none')
        saveas(gcf,fullfile(rawdata_root,['sub-',intern_cfg.entities.subject , '_ses-', intern_cfg.entities.session, '_task-',intern_cfg.entities.task, '_acq-',intern_cfg.entities.acquisition, '_run-',num2str(intern_cfg.entities.run), '_AFTER_relabeling.png']))

        %close all
    end
    
    %%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Convert data to BIDS
    [cfg,intern_cfg] = BIDS_retrieve_fieldtrip_settings(cfg, intern_cfg);

    % Now convert data to BIDS !
    if startsWith(cfg.ses,pattern('EcogLfpMedOffDys'))
        disp('Does not convert following because of MedOffDys:'); cfg.ses %take instead MedOffOnDys data
    elseif startsWith(cfg.ses,pattern('EcogLfpMedOnDys'))
        disp('Does not convert following because of MedOnDys:'); cfg.ses %take instead MedOffOnDys data
    else
        data2bids(cfg, intern_cfg.data);
        % ERROR INCONSISTENT NUMBER OF CHANNELS: then check the ECOG Model perhaps size 0 instead 6


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% save configuration data when needed
        % cd(JsonFolder)
        % % remove fields that should not be printed
        % intern_cfg_save = rmfield(intern_cfg,{'data','chs_other', 'rawdata_root'});
        % savejson('',intern_cfg_save,intern_cfg.jsonfile)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Quick fix for the scans.json file
        scans_json_fname = sprintf('sub-%s_ses-%s_scans.json',cfg.sub,cfg.ses);

        scans_json.acq_time.Description         = 'date and time of acquistion in the format YYYY-MM-DDThh:mm:ss. In case of missing timepoint format is YYYY-MM-DDT00:00:00';
        scans_json.acq_time.Units               = 'datetime';
    %    scans_json.acq_time.TermURL             = char("https:\\tools.ietf.org\html\rfc3339#section-5.6");
    %    scans_json.medication_sate.Description  = 'state of medication during recording';
    %    scans_json.medication_sate.Levels.OFF   = 'OFF parkinsonian medication';
    %    scans_json.medication_sate.Levels.ON    = 'ON parkinsonian medication';
    %    scans_json.UPDRS_III.Description        = char("Score of the unified Parkinson's disease rating scale (UPDRS) part III, as determined on the day of recording.");
    %    scans_json.UPDRS_III.TermURL            = char("https:\\doi.org\10.1002\mds.10473");

        fileID = fopen(fullfile(cfg.bidsroot,cfg.sub,cfg.ses,scans_json_fname));
        savejson('',scans_json,scans_json_fname)
        movefile(scans_json_fname,fullfile(cfg.bidsroot,['sub-' cfg.sub],['ses-' cfg.ses]))
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Quick fix for the sessions.json file
        sessions_json_fname = sprintf('sub-%s_sessions.json',cfg.sub);

        sessions_json.acq_date_no_time.Description         = 'date of acquistion in the format YYYY-MM-DDT00:00:00, not indicating time';
        sessions_json.acq_date_no_time.Units               = 'date';
        sessions_json.acq_date_no_time.TermURL             = char("https:\\tools.ietf.org\html\rfc3339#section-5.6");
        sessions_json.medication_sate.Description  = 'state of medication during recording';
        sessions_json.medication_sate.Levels.OFF   = 'OFF parkinsonian medication';
        sessions_json.medication_sate.Levels.ON    = 'ON parkinsonian medication';
        sessions_json.UPDRS_III.Description        = char("Score of the unified Parkinson's disease rating scale (UPDRS) part III, as determined on the day of recording.");
        sessions_json.UPDRS_III.TermURL            = char("https:\\doi.org\10.1002\mds.10473");
        sessions_json.subscore_tremor_right.Description        = char("Tremor subscore right of the unified Parkinson's disease rating scale (UPDRS) part III, as determined on the day of recording.");
        sessions_json.subscore_tremor_left.Description        = char("Tremor subscore left of the unified Parkinson's disease rating scale (UPDRS) part III, as determined on the day of recording.");
        sessions_json.subscore_tremor_total.Description        = char("Tremor subscore total of the unified Parkinson's disease rating scale (UPDRS) part III, as determined on the day of recording.");
        sessions_json.subscore_rigidity_right.Description        = char("Rigidity subscore right of the unified Parkinson's disease rating scale (UPDRS) part III, as determined on the day of recording.");
        sessions_json.subscore_rigidity_left.Description        = char("Rigidity subscore left of the unified Parkinson's disease rating scale (UPDRS) part III, as determined on the day of recording.");
        sessions_json.subscore_rigidity_total.Description        = char("Rigidity subscore total of the unified Parkinson's disease rating scale (UPDRS) part III, as determined on the day of recording.");
        sessions_json.subscore_bradykinesia_right.Description        = char("Bradykinesia subscore right of the unified Parkinson's disease rating scale (UPDRS) part III, as determined on the day of recording.");
        sessions_json.subscore_bradykinesia_left.Description        = char("Bradykinesia subscore left of the unified Parkinson's disease rating scale (UPDRS) part III, as determined on the day of recording.");
        sessions_json.subscore_bradykinesia_total.Description        = char("Bradykinesia subscore total of the unified Parkinson's disease rating scale (UPDRS) part III, as determined on the day of recording.");

        fileID = fopen(fullfile(cfg.bidsroot,cfg.sub,sessions_json_fname));
        savejson('',sessions_json,sessions_json_fname)
        movefile(sessions_json_fname,fullfile(cfg.bidsroot,['sub-' cfg.sub]))
    end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Make duplicates of the REST Data in case of MedOnDys or MedOffDys
    clarification_MedOffOnDys= strjoin({'Dyskinesia Protocol:',...
    'The filename indicates time after intake of a fast-acting dopaminergic agent (Madopar LT) as Dopa XX (minutes after intake).', ...
    'For all rest analyses aiming to compare OFF and ON medication states:', ...
    'use the Dopa00 or pre-medication (DopaPre) for OFF and', ...
    'use the Dopa60 (Dopa50-Dopa70) for ON medication conditions.', ...
    ' ', ...
    'A mere copy of the recordings can be found in the session folder MedOnDys and MedOffDys.', ...
    'The data is being managed in, written in and copied from the MedOffOnDys folder.', ...
    'The MedOffOnDys contains also all other recordings of that session.'},'\n');
    
    if contains(cfg.ses,'Dys')
        if contains(cfg.ses,'MedOffOnDys')
        
            time_since_medication = str2double(cfg.acq(isstrprop(cfg.acq, 'digit')));
            if isnan(time_since_medication) && (~contains(cfg.acq, 'DopaPre'))
                error('Dopa time is forgotten in acq entity')
            end

            if time_since_medication==0 || contains(cfg.acq, 'DopaPre')
                cfg.ses = strrep(cfg.ses,'MedOffOn','MedOff');                
                cfg.sessions.medication_state = strrep(cfg.sessions.medication_state,'OFFON','OFF');

                create_a_copy = 1;

            elseif 50 <= time_since_medication && time_since_medication<= 70
                cfg.ses = strrep(cfg.ses,'MedOffOn','MedOn');
                cfg.sessions.medication_state = strrep(cfg.sessions.medication_state,'OFFON','ON');

                create_a_copy = 1;
            elseif (50 <= time_since_medication && time_since_medication) && contains(cfg.task,'VigorStim')
                cfg.ses = strrep(cfg.ses,'MedOffOn','MedOn');
                cfg.sessions.medication_state = strrep(cfg.sessions.medication_state,'OFFON','ON');

                create_a_copy = 1;
            else
                create_a_copy = 0;
            end
            %need to implement that if the beh folder only exist in the medoff medon that it should be
            %copied anyhow



            if create_a_copy == 1
                UPDRS=readtable('UPDRS_Berlin.xlsx','sheet','recording_detailed');
                rownr = find(and(contains(UPDRS.Subject, cfg.sub) , contains(UPDRS.Session, cfg.ses)));
                if size(rownr)==[1,1]
                    cfg.sessions.UPDRS_III = UPDRS.UPDRS_III(rownr);
                    cfg.sessions.subscore_tremor_right = UPDRS.subscore_tremor_right(rownr);
                    cfg.sessions.subscore_tremor_left = UPDRS.subscore_tremor_left(rownr);
                    cfg.sessions.subscore_tremor_total = UPDRS.subscore_tremor_total(rownr);
                    cfg.sessions.subscore_rigidity_right = UPDRS.subscore_rigidity_right(rownr);
                    cfg.sessions.subscore_rigidity_left = UPDRS.subscore_rigidity_left(rownr);
                    cfg.sessions.subscore_rigidity_total = UPDRS.subscore_rigidity_total(rownr);
                    cfg.sessions.subscore_bradykinesia_right = UPDRS.subscore_bradykinesia_right(rownr);
                    cfg.sessions.subscore_bradykinesia_left = UPDRS.subscore_bradykinesia_left(rownr);
                    cfg.sessions.subscore_bradykinesia_total = UPDRS.subscore_bradykinesia_total(rownr);
                else
                    cfg.sessions.UPDRS_III = 'n/a';
                    cfg.sessions.subscore_tremor_right = 'n/a';
                    cfg.sessions.subscore_tremor_left = 'n/a';
                    cfg.sessions.subscore_tremor_total = 'n/a';
                    cfg.sessions.subscore_rigidity_right = 'n/a';
                    cfg.sessions.subscore_rigidity_left = 'n/a';
                    cfg.sessions.subscore_rigidity_total = 'n/a';
                    cfg.sessions.subscore_bradykinesia_right = 'n/a';
                    cfg.sessions.subscore_bradykinesia_left = 'n/a';
                    cfg.sessions.subscore_bradykinesia_total = 'n/a';
                end

                data2bids(cfg, intern_cfg.data);

                fileID = fopen('README.txt','w');
                fprintf(fileID, clarification_MedOffOnDys);
                fclose(fileID);
                movefile('README.txt',fullfile(cfg.bidsroot,['sub-' cfg.sub],['ses-' cfg.ses]))

                scans_json_fname = sprintf('sub-%s_ses-%s_scans.json',cfg.sub,cfg.ses);
                fileID = fopen(fullfile(cfg.bidsroot,cfg.sub,cfg.ses,scans_json_fname));
                savejson('',scans_json,scans_json_fname)
                movefile(scans_json_fname,fullfile(cfg.bidsroot,['sub-' cfg.sub],['ses-' cfg.ses]))


            else
                disp('Not relevant MedOffOnDys Rest recording:')
                disp([cfg.sub , ' ' , cfg.task, ' ', cfg.acq])
            end
        else
            error('mistake in session naming for dyskinesia')
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% move the config file out of the way -> inside the rawdata
    movefile( intern_cfg.jsonfile , fullfile(cfg.bidsroot) )
end
close all