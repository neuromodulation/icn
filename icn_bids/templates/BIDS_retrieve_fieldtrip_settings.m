function [cfg,intern_cfg] = BIDS_retrieve_fieldtrip_settings(cfg,intern_cfg, method)
    arguments
        cfg struct
        intern_cfg struct
        method (1,:) char {mustBeMember(method,{'readjson','update_channels','convert'})} = 'convert'
    end    

% To Do Jonathan: need to check the manufacturer to know the abbreviation and the number of
    % channels
    if strcmp(method , 'readjson')
        %% retrieve variables of the json file
        fname = intern_cfg.jsonfile; 
        fid = fopen(fname); 
        raw = fread(fid,inf); 
        str = char(raw'); 
        fclose(fid);
        temp = jsondecode(str);
        json_names=fieldnames(jsondecode(str));
        for k=1:length(json_names)
            eval(['intern_cfg.' json_names{k} '=temp.' json_names{k}]);            
        end
        return
    end

    
    

    %% assign all variables to the cfg
    DBS_target = intern_cfg.DBS_target; % explicit re-assign can remain here for the moment
    %DBS_target = 'STN';
    %DBS_target = "VIM";
    %DBS_target = "GPI";

    DBS_hemispheres = intern_cfg.DBS_hemispheres;
    if strcmp(DBS_hemispheres, 'left')
        DBS_hemispheres = {'L'};
    elseif strcmp(DBS_hemispheres, 'right')
        DBS_hemispheres = {'R'};
    elseif strcmp(DBS_hemispheres, 'bilateral')
        DBS_hemispheres = {'R', 'L'};
    end
    DBS_model=intern_cfg.DBS_model;
    %DBS_model = 'SenSight Short'; %Medtronic
    %DBS_model = 'SenSight Long'; %Medtronic
    %DBS_model = 'Vercise Cartesia X'; %Boston Scientific
    %DBS_model = 'Vercise Cartesia'; %Boston Scientific
    %DBS_model = 'Vercise Standard'; %Boston Scientific
    %DBS_model = 'Abbott Directed Long'; %Abbott
    %DBS_model = 'Abbott Directed Short'; %Abbott

    ECOG_target = intern_cfg.ECOG_target;
    %ECOG_target = 'SMC'; % Sensorimotor Cortex
    if strcmp(ECOG_target, 'SMC')
        ECOG_target_long = 'sensorimotor cortex';
    else
        error('ECOG target not found, please specify a valid target.')
    end

    if ~isfield(intern_cfg,'ECOG_hemisphere')
        intern_cfg.ECOG_hemisphere=false;
    else
        ECOG_hemisphere=intern_cfg.ECOG_hemisphere;
    end
    %ECOG_hemisphere = 'R';
    %ECOG_hemisphere = 'L';

    ECOG_model=intern_cfg.ECOG_model;
    %ECOG_model = 'TS06R-AP10X-0W6'; % manufacturer: Ad-Tech
    %ECOG_model ='DS12A-SP10X-000'; % manufacturer: Ad-Tech


    hardware_manufacturer   = 'TMSi';

    % Handle DBS lead model

    if strcmp(DBS_model, 'SenSight Short')
        DBS_contacts           = 8;
        DBS_manufacturer       = 'Medtronic';
        DBS_manufacturer_short = "MT";
        DBS_description        = '8-contact, 4-level, directional DBS lead. 0.5 mm spacing.';
        DBS_material           = 'platinum/iridium';    
        DBS_directional        = 'yes';
    elseif strcmp(DBS_model, 'SenSight Long')
        DBS_contacts           = 8;
        DBS_manufacturer       = 'Medtronic';
        DBS_manufacturer_short = "MT";
        DBS_description        = '8-contact, 4-level, directional DBS lead. 1.5 mm spacing.';
        DBS_material           = 'platinum/iridium';
        DBS_directional        = 'yes';
    elseif strcmp(DBS_model, 'Vercise Cartesia X')
        DBS_contacts           = 16;
        DBS_manufacturer       = 'Boston Scientific';
        DBS_manufacturer_short = "BS";
        DBS_description        = '16-contact, 5-level, directional DBS lead. 0.5 mm spacing.';
        DBS_material           = 'platinum/iridium';
        DBS_directional        = 'yes';
    elseif strcmp(DBS_model, 'Vercise Cartesia')
        DBS_contacts           = 8;
        DBS_manufacturer       = 'Boston Scientific';
        DBS_manufacturer_short = "BS";
        DBS_description        = '8-contact, 4-level, directional DBS lead. 0.5 mm spacing.';
        DBS_material           = 'platinum/iridium';
        DBS_directional        = 'yes';
    elseif strcmp(DBS_model, 'Vercise Standard')
        DBS_contacts           = 8;
        DBS_manufacturer       = 'Boston Scientific';
        DBS_manufacturer_short = "BS";
        DBS_description        = '8-contact, 8-level, non-directional DBS lead. 0.5 mm spacing.';
        DBS_material           = 'platinum/iridium';
        DBS_directional        = 'no';
    elseif strcmp(DBS_model, 'Abbott Directed Long')
        DBS_contacts           = 8;
        DBS_manufacturer       = 'Abbott/St Jude';
        DBS_manufacturer_short = "AB";
        DBS_description        = '8-contact, 4-level, directional DBS lead. 1.5 mm spacing.';
        DBS_material           = 'platinum/iridium';
        DBS_directional        = 'yes';
    elseif strcmp(DBS_model, 'Abbott Directed Short')
        DBS_contacts           = 8;
        DBS_manufacturer       = 'Abbott/St Jude';
        DBS_manufacturer_short = "AB";
        DBS_description        = '8-contact, 4-level, directional DBS lead. 0.5 mm spacing.';
        DBS_material           = 'platinum/iridium';
        DBS_directional        = 'yes';
    else
        error('DBS model not found, please specify a valid DBS lead.')
    end

    chs_DBS = cell(DBS_contacts,1);
    for i = 1:length(DBS_hemispheres)
        hemisphere = DBS_hemispheres{i};
        for ind = 1:DBS_contacts
            items = ["LFP", hemisphere, string(ind), DBS_target, DBS_manufacturer_short];
            ch_name = join(items, '_');
            chs_DBS{ind+DBS_contacts*(i-1)} = ch_name{1};
        end
    end

    % Handle ECOG electrode model
    if strcmp(ECOG_model, 'TS06R-AP10X-0W6')
        ECOG_contacts              = 6;
        ECOG_manufacturer_short    = "AT";
        ECOG_manufacturer          = 'Ad-Tech';
        ECOG_location              = 'subdural';
        ECOG_material              = 'platinum';
        ECOG_description           = '6-contact, 1x6 narrow-body long term monitoring strip. Platinum contacts, 10mm spacing, contact size 4.0 mm diameter/1.8 mm exposure.';
    elseif strcmp(ECOG_model, 'DS12A-SP10X-000')
        ECOG_contacts              = 12;
        ECOG_manufacturer_short    = "AT";
        ECOG_manufacturer          = 'Ad-Tech';
        ECOG_location              = 'subdural';
        ECOG_material              = 'platinum';
        ECOG_description           = '12-contact, 1x6 dual sided long term monitoring strip. Platinum contacts, 10mm spacing, contact size 4.0 mm diameter/2.3 mm exposure. Platinum marker.';

    else
        error('ECOG model not found, please specify a valid ECOG electrode.')
    end

    chs_ECOG = cell(ECOG_contacts,1);
    for ind = 1:ECOG_contacts
        items = ["ECOG", ECOG_hemisphere, string(ind), ECOG_target, ECOG_manufacturer_short];
        ch_name = join(items, '_');
        chs_ECOG{ind} = ch_name{1};
    end

    chs_final = [chs_DBS; chs_ECOG; intern_cfg.chs_other];
    
    
    if strcmp(DBS_directional, 'yes')
        directional = 'directional';
    else
        directional = 'non-directional';
    end

    %% Now assign channel types

    intern_cfg.data.label          = chs_final;
    intern_cfg.data.hdr.nChans     = length(chs_final);
    intern_cfg.data.hdr.label      = intern_cfg.data.label;
    % Set channel types and channel units
    chantype            = cell(intern_cfg.data.hdr.nChans,1);
    for ch = 1:intern_cfg.data.hdr.nChans
        if contains(chs_final{ch},'LFP')
            chantype(ch) = {'DBS'};
        elseif contains(chs_final{ch},'ECOG')
            chantype(ch) = {'ECOG'};
        elseif contains(chs_final{ch},'EEG')
            chantype(ch) = {'EEG'};
        elseif contains(chs_final{ch},'EMG')
            chantype(ch) = {'EMG'};    
        elseif contains(chs_final{ch},'ECG')
            chantype(ch) = {'ECG'};   
        else
            chantype(ch) = {'MISC'};   
        end
    end

   intern_cfg.data.hdr.chantype   = chantype;
    intern_cfg.data.hdr.chanunit   = repmat({'uV'}, intern_cfg.data.hdr.nChans,1);
        
        
    if strcmp(method , 'update_channels')
        intern_cfg.chs_final = chs_final;
        return
    end

    
    %% Note which channels were bad and why
    %bad = {'LFP_L_7_STN_MT' 'LFP_L_8_STN_MT' 'LFP_L_9_STN_MT' 'LFP_L_16_STN_MT' 'LFP_R_7_STN_MT' 'LFP_R_8_STN_MT' 'LFP_R_9_STN_MT'};
    %why = {'Stimulation contact' 'Stimulation contact' 'Stimulation contact' 'Reference electrode' 'Stimulation contact' 'Stimulation contact' 'Stimulation contact' 'Stimulation contact'};
%     bad ={'LFP_L_8_STN_MT'};
%     why = {'Reference electrode'};
    bad = intern_cfg.bad;
    why = intern_cfg.why;
    iEEGRef = intern_cfg.iEEGRef;
    %iEEGRef = 'LFP_L_8_STN_BS'; % what is the reference contact?

    bads = repmat({'good'},intern_cfg.data.hdr.nChans,1);
    bads_descr = repmat({'n/a'},intern_cfg.data.hdr.nChans,1);
    for k=1:length(chs_final)
        if ismember(chs_final{k}, bad)
            bads{k} = 'bad';
            bads_descr{k} = why{find(strcmp(bad,chs_final{k}))};
        end
    end

    %% Initalize containers for BIDS conversion
    keySet = {'Rest', 'UPDRSIII', 'SelfpacedRotationL','SelfpacedRotationR',...
        'BlockRotationL','BlockRotationR', 'Evoked', 'SelfpacedSpeech',...
        'ReadRelaxMoveL', 'VigorStimR', 'VigorStimL', 'SelfpacedHandTapL',...
        'SelfpacedHandTapR', 'Free'...
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
        'Selfpaced left hand tapping, circa every 10 seconds, without counting, in resting seated position.',...
        'Selfpaced right hand tapping, circa every 10 seconds, without counting, in resting seated position.',...
        'Free period, no instructions, this period is recorded during the Dyskinesia-Protocol to monitor the increasing Dopamine-Level'...
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
        'Keep both hands resting on your legs, and tap with your left hand by raising the hand and fingers of your left hand, without letting the arm be lifted from the leg. Do not count in between rotations.',...
        'Keep both hands resting on your legs, and tap with your right hand by raising the hand and fingers of your right hand, without letting the arm be lifted from the leg. Do not count in between rotations.',...
        'Free period, without instructions or restrictions, between Rest-measurement and Task-measurements'...
        };
    %task_descr = containers.Map(keySet,descrSet);
    %task_instr = containers.Map(keySet,instructionSet);
    task_descr = intern_cfg.task_description;
    task_instr = intern_cfg.task_instructions;

    %% Now write data to BIDS

    % Initialize a cell array of 'n/a' for practicality
    n_a = repmat({'n/a'},intern_cfg.data.hdr.nChans,1);

    % adept for each different recording
    cfg = [];
    cfg.method                  = 'convert';
    cfg.bidsroot                = intern_cfg.rawdata_root;
    cfg.datatype                = 'ieeg';
    cfg.sub                     = intern_cfg.subject;%'009';
    cfg.ses                     = intern_cfg.session;
    cfg.task                    = intern_cfg.task;
    cfg.acq                     = intern_cfg.acquisition; %'StimOff01';  % add here 'Dopa00' during dyskinesia-protocol recording: e.g. 'StimOff01Dopa30'. (Dyskinesia-protocol recordings start at the intake of an higher than normal Levodopa-dosage, and will always be labeled MedOn)
    cfg.run                     = intern_cfg.run;
    cfg.space                   = intern_cfg.space; %'MNI152NLin2009bAsym';

    % Provide info for the scans.tsv file
    % the acquisition time could be found in the folder name of the recording

    cfg.scans.acq_time              =  intern_cfg.time_of_acquisition;%'2022-01-24T09:36:27';
    if contains(cfg.ses, 'Off')
        cfg.scans.medication_state  = 'OFF';
    else
        cfg.scans.medication_state  = 'ON';
    end
    cfg.scans.UPDRS_III             =  intern_cfg.UPDRS_session;%'n/a'; % need to be calcuated.

    % Specify some general information
    cfg.InstitutionName                         = 'Charite - Universitaetsmedizin Berlin, corporate member of Freie Universitaet Berlin and Humboldt-Universitaet zu Berlin, Department of Neurology with Experimental Neurology/BNIC, Movement Disorders and Neuromodulation Unit';
    cfg.InstitutionAddress                      = 'Chariteplatz 1, 10117 Berlin, Germany';
    cfg.dataset_description.Name                = 'BIDS_Berlin_ECOG_LFP';
    cfg.dataset_description.BIDSVersion         = '1.6.0';
    cfg.dataset_description.License             = 'n/a';
    cfg.dataset_description.Funding             = {'Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - Project-ID 424778381 - TRR 295'};
    cfg.dataset_description.Authors             = {'Johannes Busch', 'Meera Chikermane', 'Katharina Faust', 'Lucia Feldmann', 'Jeroen Habets', 'Richard Koehler', 'Andrea Kuehn', 'Roxanne Lofredi', 'Timon Merk', 'Wolf-Julian Neumann', 'Gerd-Helge Schneider', 'Ulrike Uhlig', 'Jonathan Vanhoecke'};
    cfg.dataset_description.Acknowledgements    = 'Special thanks to all other people involved in acquiring the data.';


    % Provide the long description of the task and participant instructions
    cfg.TaskName                = cfg.task;
    cfg.TaskDescription         = intern_cfg.task_description ;%task_descr(cfg.task);
    cfg.Instructions            = intern_cfg.task_instructions ;%task_instr(cfg.task);

    % Provide info about recording hardware

    if strcmp(hardware_manufacturer,'TMSi')
        cfg.Manufacturer                = 'Twente Medical Systems International B.V. (TMSi)';
        cfg.ManufacturersModelName      = 'Saga 64+';
        cfg.SoftwareVersions            = 'TMSi Polybench - QRA for SAGA - REV1.1.0';
        cfg.DeviceSerialNumber          = '1005190056';
        cfg.channels.low_cutoff         = repmat({'0'},intern_cfg.data.hdr.nChans,1);
        cfg.channels.high_cutoff        = repmat({'2100'},intern_cfg.data.hdr.nChans,1); 
        cfg.channels.high_cutoff(contains(string(chs_final),["LFP", "ECOG","EEG"])) = {'1600'};
        Hardware_Filters.Anti_AliasFilter.Low_Pass.UnipolarChannels     = 1600;
        Hardware_Filters.Anti_AliasFilter.Low_Pass.BipolarChannels      = 2100;
        Hardware_Filters.Anti_AliasFilter.Low_Pass.AuxiliaryChannels    = 2100;
    %elseif strcmp(hardware_manufacturer,'Alpha Omega')
    %elseif strcmp(hardware_manufacturer,'Newronika')
    else
        error('Please define a valid hardware manufacturer')
    end

    % need to check in the LFP excel sheet on the S-drive
    % Provide info about the participant	
    cfg.participants.sex                    = intern_cfg.sex; %found in the clinical data (you probably don't have have access to the SAP, sometimes it can be found in AG-Bewegungsstörungen/Filme)
    cfg.participants.handedness             = intern_cfg.handedness; %LFP excel sheet
    cfg.participants.age                    = intern_cfg.age ; %LFP excel sheet
    cfg.participants.date_of_implantation   = intern_cfg.date_of_implantation ;  %'2022-01-20T00:00:00'; %LFP excel sheet
    cfg.participants.UPDRS_III_preop_OFF    = 'n/a';%-> UPDRS is on session level
    cfg.participants.UPDRS_III_preop_ON     = 'n/a';%-> UPDRS is on session level
    cfg.participants.disease_duration       = intern_cfg.disease_duration;%7; %LFP excel sheet
    cfg.participants.PD_subtype             = intern_cfg.PD_subtype; %'akinetic-rigid'; %SAP
    cfg.participants.symptom_dominant_side  = intern_cfg.symptom_dominant_side;% 'right'; %LFP excel sheet
    cfg.participants.LEDD                   = intern_cfg.LEDD; %1600; %calculated from lab book with https://www.parkinsonsmeasurement.org/toolBox/levodopaEquivalentDose.htm
    cfg.participants.DBS_target                 = DBS_target;
    cfg.participants.DBS_manufacturer           = DBS_manufacturer;
    cfg.participants.DBS_model                  = DBS_model;
    cfg.participants.DBS_directional            = DBS_directional;
    cfg.participants.DBS_contacts               = DBS_contacts;
    cfg.participants.DBS_description            = DBS_description;

    % Info about the ECOG electrode
    cfg.participants.ECOG_target                = ECOG_target_long;
    
    if strcmp('R', ECOG_hemisphere)
        cfg.participants.ECOG_hemisphere            = 'right';
    else
        cfg.participants.ECOG_hemisphere            = 'left';
    end
    cfg.participants.ECOG_manufacturer          = ECOG_manufacturer;
    cfg.participants.ECOG_model                 = ECOG_model;
    cfg.participants.ECOG_location              = ECOG_location;
    cfg.participants.ECOG_material              = ECOG_material;
    cfg.participants.ECOG_contacts              = ECOG_contacts;
    cfg.participants.ECOG_description           = ECOG_description;
    

    


    % Provide info for the coordsystem.json file
    % remains always the same in berlin
    cfg.coordsystem.IntendedFor                         = "n/a"; % OPTIONAL. Path or list of path relative to the subject subfolder pointing to the structural MRI, possibly of different types if a list is specified, to be used with the MEG recording. The path(s) need(s) to use forward slashes instead of backward slashes (e.g. "ses-<label>/anat/sub-01_T1w.nii.gz").
    cfg.coordsystem.iEEGCoordinateSystem                = cfg.space; % REQUIRED. Defines the coordinate system for the iEEG electrodes. See Appendix VIII for a list of restricted keywords. If positions correspond to pixel indices in a 2D image (of either a volume-rendering, surface-rendering, operative photo, or operative drawing), this must be "Pixels". For more information, see the section on 2D coordinate systems
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
    sens.label = chs_final(contains(string(chs_final),["LFP", "ECOG"]));

    % Electrode positions are imported from external files (e.g. Lead-DBS
    % ea_reconstruction.mat) and a sens FieldTrip struct is created 
    % (see FT_DATATYPE_SENS)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % To Do: -> for now fixed, if else statement fixed for models: Cartesia X and Sensight
    % this is the channel position for medtronic
    sens.chanpos = [
        zeros(DBS_contacts, 3); ...
        zeros(DBS_contacts, 3); ...
        zeros(ECOG_contacts, 3)]; %what is this 12 refering to? I replaced it with n_ECOG_contact 
    % this is for medtronic
    if isfield(intern_cfg,'ECOG_localization')
        sens.chanpos(DBS_contacts+DBS_contacts + 1 : end,1:3) = intern_cfg.ECOG_localization;
    end
    
    % define size of single electrode contacts
    % if 1 DBS contact per level: size=6; if 3 contacts per level: s=1.5
    if isfield(cfg.participants.DBS_model(:,8),'SenSight')
        cfg.electrodes.size = {
            6 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
            6 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
            4.15 4.15 4.15 4.15 4.15 4.15}; %  ECoG contacts std 4.15
    
    elseif isfield(cfg.participants.DBS_model, 'Vercise Cartesia X')
        cfg.electrodes.size = {
            1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
            1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
            4.15 4.15 4.15 4.15 4.15 4.15 ...
            };
    end
    % sens.chanpos = [
    %     zeros(DBS_contacts, 3); ...
    %     zeros(DBS_contacts, 3); ...
    %     zeros(6, 3)]; %what is this 6 refering to?
    sens.elecpos = sens.chanpos;
    cfg.elec     = sens;
    cfg.electrodes.name = sens.label;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % RECOMMENDED. Material of the electrode, e.g., Tin, Ag/AgCl, Gold
    cfg.electrodes.material     = [
        repmat({DBS_material},DBS_contacts*2,1); 
        repmat({ECOG_material}, ECOG_contacts ,1)];
    cfg.electrodes.manufacturer = [
        repmat({cfg.participants.DBS_manufacturer},DBS_contacts*2,1);
        repmat({cfg.participants.ECOG_manufacturer},6,1)];
    cfg.electrodes.group        = [
        repmat({'DBS_right'},DBS_contacts,1);
        repmat({'DBS_left'},DBS_contacts,1);
        repmat({['ECOG_' cfg.participants.ECOG_hemisphere]},ECOG_contacts,1)];
    cfg.electrodes.hemisphere   = [
        repmat({'R'},DBS_contacts,1);
        repmat({'L'},DBS_contacts,1);
        repmat({ECOG_hemisphere},ECOG_contacts,1)];
    % RECOMMENDED. Type of the electrode (e.g., cup, ring, clip-on, wire, needle)
    cfg.electrodes.type         = [  
        repmat({'depth'},DBS_contacts*2,1); %=> this is DBS
        repmat({'strip'},ECOG_contacts,1)]; %=> this is the ECOG

    % RECOMMENDED. Impedance of the electrode in kOhm
    cfg.electrodes.impedance    = repmat({'n/a'},length(sens.label),1);  
    cfg.electrodes.dimension    = [  
        repmat({sprintf('[1x%d]',DBS_contacts)},DBS_contacts*2,1);
        repmat({sprintf('[1x%d]',ECOG_contacts)},ECOG_contacts,1)];

    % Provide special channel info
    cfg.channels.name               = chs_final;
    cfg.channels.type               = chantype;
    cfg.channels.units              = intern_cfg.data.hdr.chanunit;
    sf = cell(length(chs_final),1);
    sf(:) = {intern_cfg.data.fsample};
    cfg.channels.sampling_frequency = sf;
    %cfg.channels.description       = ft_getopt(cfg.channels, 'description'        , nan);  % OPTIONAL. Brief free-text description of the channel, or other information of interest. See examples below.

    % Reference channels

    cfg.ieeg.iEEGReference = iEEGRef;
    typeSet = {'EEG', 'ECOG', 'DBS', 'SEEG', 'EMG', 'ECG', 'MISC'};
    refSet = {iEEGRef, iEEGRef, iEEGRef, iEEGRef, 'bipolar', 'bipolar', 'n/a'};
    ref_map = containers.Map(typeSet,refSet);
    cfg.channels.reference = arrayfun(@(ch_type) {ref_map(ch_type{1})}, chantype);
    % always notch filter on n/a
    cfg.channels.notch              = n_a;

    cfg.channels.group              = n_a;
    cfg.channels.group(startsWith(cfg.channels.name, 'LFP_R')) = {'DBS_right'};
    cfg.channels.group(startsWith(cfg.channels.name, 'LFP_L')) = {'DBS_left'};
    cfg.channels.group(startsWith(cfg.channels.name, 'ECOG_R')) = {'ECOG_right'};
    cfg.channels.group(startsWith(cfg.channels.name, 'ECOG_L')) = {'ECOG_left'};
    cfg.channels.group(startsWith(cfg.channels.name, 'EEG')) = {'EEG'};
    cfg.channels.group(startsWith(cfg.channels.name, 'EMG_L')) = {'EMG_left'};
    cfg.channels.group(startsWith(cfg.channels.name, 'EMG_R')) = {'EMG_right'};
    cfg.channels.group(startsWith(cfg.channels.name, 'ECG')) = {'ECG'};
    cfg.channels.group(startsWith(cfg.channels.name, 'ACC_L')) = {'accelerometer_left'};
    cfg.channels.group(startsWith(cfg.channels.name, 'ACC_R')) = {'accelerometer_right'};
    cfg.channels.group(startsWith(cfg.channels.name, 'ANALOG_L_ROTA')) = {'rotameter_left'};
    cfg.channels.group(startsWith(cfg.channels.name, 'ANALOG_R_ROTA')) = {'rotameter_right'};

    cfg.channels.description        = n_a; %the descriptions below are matching those of MNE despite the odd spelling!
    cfg.channels.description(startsWith(cfg.channels.name, 'LFP')) = {'Deep Brain Stimulation'};
    cfg.channels.description(startsWith(cfg.channels.name, 'ECOG')) = {'Electrocorticography'};
    cfg.channels.description(startsWith(cfg.channels.name, 'EEG')) = {'ElectroEncephaloGram'};
    cfg.channels.description(startsWith(cfg.channels.name, 'EMG')) = {'Electromyography'};
    cfg.channels.description(startsWith(cfg.channels.name, 'ECG')) = {'ElectroCardioGram'};
    cfg.channels.description(startsWith(cfg.channels.name, 'EOG')) = {'ElectroOculoGram'};
    cfg.channels.description(startsWith(cfg.channels.name, 'ACC')) = {'Accelerometer'};
    cfg.channels.description(startsWith(cfg.channels.name, 'MISC')) = {'Miscellaneous'};
    cfg.channels.description(startsWith(cfg.channels.name, 'STIM')) = {'Trigger'};
    cfg.channels.description(startsWith(cfg.channels.name, 'ANALOG_L_ROTA')) = {'Rotameter'};
    cfg.channels.description(startsWith(cfg.channels.name, 'ANALOG_R_ROTA')) = {'Rotameter'};


    cfg.channels.status             = bads;
    cfg.channels.status_description = bads_descr;

    % these are iEEG specific
    cfg.ieeg.PowerLineFrequency     = 50;   % since recorded in the Europe
    cfg.ieeg.iEEGGround             = 'Right shoulder patch';
    % this is to be specified for each model
    cfg.ieeg.iEEGPlacementScheme    = 'Left subdural cortical strip and bilateral subthalamic nucleus (STN) deep brain stimulation (DBS) leads.';

    format_groups = 'ECOG_%s: %d-contact, 1x%d dual-sided long-term monitoring %s strip on %s. DBS_left: 1x%d %s %s DBS lead in left %s, DBS_right: 1x%d %s %s DBS lead in right %s.';
    cfg.ieeg.iEEGElectrodeGroups = sprintf(format_groups, cfg.participants.ECOG_hemisphere, ECOG_contacts, ECOG_contacts, ECOG_manufacturer, ECOG_target_long,...
        DBS_contacts,  DBS_manufacturer, DBS_model, DBS_target,...
        DBS_contacts,  DBS_manufacturer, DBS_model, DBS_target );
    % cfg.ieeg.iEEGElectrodeGroups    = 'ECOG_strip: 6-contact, 1x6 dual sided long term monitoring AdTech strip on left sensorimotor cortex, DBS_left: 1x16 Boston Scientific directional DBS lead (Cartesia X) in left STN, DBS_right: 1x16 Boston Scientific directional DBS lead (Cartesia X) in right STN.';
    
    % To Do: Software filters => need to check which were used
    % eg. {"Anti-aliasing filter": {"half-amplitude cutoff (Hz)": 500, "Roll-off": "6dB/Octave"}}.
    cfg.ieeg.SoftwareFilters        = 'no additional filters'; %MUST
    cfg.ieeg.HardwareFilters        = Hardware_Filters; %Recommended
    cfg.ieeg.RecordingType          = 'continuous';
    if contains(cfg.acq, 'On')
        cfg.ieeg.ElectricalStimulation  = true;
    else
        cfg.ieeg.ElectricalStimulation  = false;
    end
    if cfg.ieeg.ElectricalStimulation
        % Enter EXPERIMENTAL stimulation settings
        % these need to be written in the lab book
        exp.DateOfSetting           = intern_cfg.stim.DateOfSetting; %"2021-11-11"
        exp.StimulationTarget       = DBS_target;
        exp.StimulationMode         = "continuous";
        exp.StimulationParadigm     = "continuous stimulation";
        exp.SimulationMontage       = "monopolar";
        L.AnodalContact             = "Ground";
        L.CathodalContact           = intern_cfg.stim.L.CathodalContact;
        L.AnodalContactDirection      = "none";
        L.CathodalContactDirection    = "omni";
        L.CathodalContactImpedance    = "n/a";
        L.StimulationAmplitude        = intern_cfg.stim.L.StimulationAmplitude;
        L.StimulationPulseWidth       = 60;
        L.StimulationFrequency        = intern_cfg.stim.L.StimulationFrequency;
        L.InitialPulseShape           = "rectangular";
        L.InitialPulseWidth           = 60;
        L.InitialPulseAmplitude       = -1.0*L.StimulationAmplitude;
        L.InterPulseDelay             = 0;
        L.SecondPulseShape            = "rectangular";
        L.SecondPulseWidth            = 60;
        L.SecondPulseAmplitude        = L.StimulationAmplitude;
        L.PostPulseInterval           = "n/a";
        exp.Left                    = L;
        R.AnodalContact             = "Ground";
        R.CathodalContact           = intern_cfg.stim.R.CathodalContact;
        R.AnodalContactDirection      = "none";
        R.CathodalContactDirection    = "omni";
        R.CathodalContactImpedance    = "n/a";
        R.StimulationAmplitude        = intern_cfg.stim.R.StimulationAmplitude;
        R.StimulationPulseWidth       = 60;
        R.StimulationFrequency        = intern_cfg.stim.R.StimulationFrequency;
        R.InitialPulseShape           = "rectangular";
        R.InitialPulseWidth           = 60;
        R.InitialPulseAmplitude       = -1.0*R.StimulationAmplitude;
        R.InterPulseDelay             = 0;
        R.SecondPulseShape            = "rectangular";
        R.SecondPulseWidth            = 60;
        R.SecondPulseAmplitude        = R.StimulationAmplitude;
        R.PostPulseInterval           = "n/a";
        exp.Right                     = R;

        % Enter CLINICAL stimulation settings (are here equal to
        % stimsettings)
%         clin.DateOfSetting           = intern_cfg.stim.DateOfSetting;
%         clin.StimulationTarget       = DBS_target;
%         clin.StimulationMode         = "continuous";
%         clin.StimulationParadigm     = "continuous stimulation";
%         clin.SimulationMontage       = "monopolar";
% %         clear L R;
% %         L                           = "OFF";
%         clin.Left                    = L;
% %         R.AnodalContact             = "G";
% %         R.CathodalContact           = "2, 3 and 4";
% %         R.AnodalContactDirection      = "none";
% %         R.CathodalContactDirection    = "omni";
% %         R.CathodalContactImpedance    = "n/a";
% %         R.StimulationAmplitude        = 1.5;
% %         R.StimulationPulseWidth       = 60;
% %         R.StimulationFrequency        = 130;
% %         R.InitialPulseShape           = "rectangular";
% %         R.InitialPulseWidth           = 60;
% %         R.InitialPulseAmplitude       = -1.5;
% %         R.InterPulseDelay             = 0;
% %         R.SecondPulseShape            = "rectangular";
% %         R.SecondPulseWidth            = 60;
% %         R.SecondPulseAmplitude        = 1.5;
% %         R.PostPulseInterval           = "n/a";
%         clin.Right                    = R;

        param.BestClinicalSetting                = "Berlin parameter preset";
        param.CurrentExperimentalSetting         = exp;
        cfg.ieeg.ElectricalStimulationParameters = param;
    end





end