function [cfg,intern_cfg] = BIDS_retrieve_fieldtrip_settings(cfg,intern_cfg, method)
arguments
    cfg struct
    intern_cfg struct
    method (1,:) char {mustBeMember(method,{'readjson','update_channels','convert'})} = 'convert'
end

% To Do Jojo: need to check the manufacturer to know the abbreviation and the number of
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
        eval(['intern_cfg.' json_names{k} '=temp.' json_names{k} ';']);
    end

    try
        if ~isempty(intern_cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.CathodalContact)
            intern_cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.CathodalContact=string(intern_cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.CathodalContact);
        end
    end
    try
        if ~isempty(intern_cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.CathodalContact)
            intern_cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.CathodalContact=string(intern_cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.CathodalContact);
        end
    end

    return;
end

%% assign all variables to the cfg
DBS_target = intern_cfg.participants.DBS_target; % explicit re-assign can remain here for the moment
%DBS_target = 'STN';
%DBS_target = 'VIM';
%DBS_target = 'GPI';

DBS_hemisphere = intern_cfg.participants.DBS_hemisphere;
if strcmp(DBS_hemisphere, 'left')
    DBS_hemispheres = {'L'};
elseif strcmp(DBS_hemisphere, 'right')
    DBS_hemispheres = {'R'};
elseif strcmp(DBS_hemisphere, 'bilateral')
    DBS_hemispheres = {'R', 'L'};
end
DBS_model=intern_cfg.participants.DBS_model;
updateDBSdevice = 0;
%DBS_model = 'SenSight Short'; %Medtronic
%DBS_model = 'SenSight Long'; %Medtronic
%DBS_model = 'Vercise Cartesia X'; %Boston Scientific
%DBS_model = 'Vercise Cartesia' %Boston Scientific
%DBS_model = 'Vercise Standard'; %Boston Scientific
%DBS_model = 'Abbott Directed Long'; %Abbott
%DBS_model = 'Abbott Directed Short'; %Abbott
%updateDBSdevice =1

ECOG_target_long = intern_cfg.participants.ECOG_target;
%ECOG_target = 'SMC'; % Sensorimotor Cortex
if strcmp(ECOG_target_long, 'sensorimotor cortex')
    ECOG_target = 'SMC';
elseif strcmp(ECOG_target_long, 'n/a')
    ECOG_target = 'n/a';
elseif strcmp(ECOG_target_long, '')
    ECOG_target = 'n/a';
else
    error('ECOG target not found, please specify a valid target.')
end

if ~isfield(intern_cfg.participants,'ECOG_hemisphere')
    %intern_cfg.participants
    ECOG_hemisphere='n/a';
elseif strcmp(intern_cfg.participants.ECOG_hemisphere,'n/a')
    %intern_cfg.participants.ECOG_hemisphere=false;
    ECOG_hemisphere='n/a';
elseif ~intern_cfg.participants.ECOG_hemisphere
    %intern_cfg.participants.ECOG_hemisphere=false;
    ECOG_hemisphere='n/a';
else
    ECOG_hemisphere=intern_cfg.participants.ECOG_hemisphere;
    if strcmp(ECOG_hemisphere, 'left')
        ECOG_hemispheres = {'L'};
    elseif strcmp(ECOG_hemisphere, 'right')
        ECOG_hemispheres = {'R'};
    elseif strcmp(ECOG_hemisphere, 'bilateral')
        ECOG_hemispheres = {'R', 'L'};
    else
        error('define a valid ECOG hemisphere')
    end
end

ECOG_model=intern_cfg.participants.ECOG_model;
%ECOG_model = 'TS06R-AP10X-0W6'; % manufacturer: Ad-Tech
%ECOG_model = 'DS12A-SP10X-000'; % manufacturer: Ad-Tech

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
elseif strcmp(DBS_model, 'n/a')
    DBS_contacts           = str2num(intern_cfg.participants.DBS_contacts);
    DBS_manufacturer       = intern_cfg.participants.DBS_manufacturer;
    DBS_manufacturer_short = "MT";
    DBS_description        = intern_cfg.participants.DBS_description;
    DBS_material           = 'platinum/iridium';
    DBS_directional        = intern_cfg.participants.DBS_directional;
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
    ECOG_manufacturer_short    = 'AT';
    ECOG_manufacturer          = 'Ad-Tech';
    ECOG_location              = 'subdural';
    ECOG_material              = 'platinum';
    ECOG_description           = '6-contact, 1x6 narrow-body long term monitoring strip. Platinum contacts, 10mm spacing, contact size 4.0 mm diameter/1.8 mm exposure.';
elseif strcmp(ECOG_model, 'DS12A-SP10X-000')
    ECOG_contacts              = 12;
    ECOG_manufacturer_short    = 'AT';
    ECOG_manufacturer          = 'Ad-Tech';
    ECOG_location              = 'subdural';
    ECOG_material              = 'platinum';
    ECOG_description           = '12-contact, 1x12 dual sided long term monitoring strip. Platinum contacts, 10mm spacing, contact size 4.0 mm diameter/2.3 mm exposure.';
elseif strcmp(ECOG_model, 'n/a') || strcmp(ECOG_model, '')
    ECOG_model                 = 'n/a';
    ECOG_contacts              = 0;
    ECOG_manufacturer_short    = 'n/a';
    ECOG_manufacturer          = 'n/a';
    ECOG_location              = 'n/a';
    ECOG_material              = 'n/a';
    ECOG_description           = 'n/a';
else
    error('ECOG model not found, please specify a valid ECOG electrode.')
end

if strcmp(DBS_directional, 'yes')
    directional = 'directional';
else
    directional = 'non-directional';
end



%% update + RESET the channels with new channels from meta data intern_cfg.channels_tsv.name

if strcmp(method , 'update_channels')
    if isfield(intern_cfg,'poly5')
        remove = [];
        %%% add new empty channels
        if length(intern_cfg.data.label)<length(intern_cfg.poly5.old)
            warning('adding empty channels, new old poly5 added')
            n_channels_to_add=length(intern_cfg.poly5.old)-length(intern_cfg.data.label);
            intern_cfg.data.label(end+1:end+n_channels_to_add) = intern_cfg.poly5.old(end+1-n_channels_to_add:end);
            intern_cfg.data.trial{1}(end+1:end+n_channels_to_add,:) = 0;
            intern_cfg.data.hdr.chanunit(end+1:end+n_channels_to_add) = {'uV'};
            intern_cfg.data.hdr.chantype(end+1:end+n_channels_to_add) = {'eeg'};
        end
        for i= 1:length(intern_cfg.data.label)
            % this maps the old channel name in the poly5 dictionary to the new channel name
            % note, as matlab does not allow spaces and hyphens in a
            % struct, the new and old names are to be found in a list
            % only select the first 15 characters, because that is how
            % fieldtrip works
            index = find(contains(intern_cfg.poly5.old,intern_cfg.data.label{i}(1:min(15, length(intern_cfg.data.label{i})))));
            if length(index) > 1
                % try to find a unique match
                control = index;
                index = find(ismember(intern_cfg.poly5.old,intern_cfg.data.label{i}(1:min(15, length(intern_cfg.data.label{i})))));
            end
            if isempty(index)
                % if no match was found, the new name is probably on
                % same position of the old one
                index = i;
                if sum(ismember(control,index))==1
                    fprintf('Channel was not found explicitly. %s is now replaced by %s \n',intern_cfg.data.label{i}, intern_cfg.poly5.new{index});
                else
                    error('The channel name matching did not work')
                end
            end

            % intern_cfg.data.label is the final channel lists, but contains here the channels
            % that need to be removed as well
            intern_cfg.data.label{i} = intern_cfg.poly5.new{index};

            % assert a good channel renaming
            %
            % extra room for assertions
            % extra room for assertions
            % extra room for assertions
            
            if updateDBSdevice %temp hack
                if startsWith(intern_cfg.data.label{i},'LFP')
                    intern_cfg.data.label{i}(end-1:end) =  char(DBS_manufacturer_short);
                end
            end

            if startsWith(intern_cfg.data.label{i},'LFP')
                assert(endsWith(intern_cfg.data.label{i}, DBS_manufacturer_short))
            elseif startsWith(intern_cfg.data.label{i},'ECOG')
                assert(endsWith(intern_cfg.data.label{i}, ECOG_manufacturer_short))
                if length(ECOG_hemispheres) == 1
                    assert(startsWith(intern_cfg.data.label{i},['ECOG_', ECOG_hemispheres{1}]))
                end
            end


            if isempty(intern_cfg.data.label{i})
                remove(end+1) = i;
            end
        end

        intern_cfg.data.label(remove) = [];
        intern_cfg.data.trial{1}(remove,:) = [];
        intern_cfg.data.hdr.chanunit(remove) = [];
        intern_cfg.data.hdr.chantype(remove) = [];
        intern_cfg.data.hdr.nChans     = length(intern_cfg.channels_tsv.name); %update the channel numbers
    else
        intern_cfg.data.label = intern_cfg.channels_tsv.name;
    end

    chs_unsorted = intern_cfg.data.label;

    %% sort the channel names
    % first sort by LFP then ECOG, then the other
    temp_LFP_R=sort(chs_unsorted(startsWith(chs_unsorted,{'LFP_R'})));
    temp_LFP_L=sort(chs_unsorted(startsWith(chs_unsorted,{'LFP_L'})));
    temp_ECOG_R=sort(chs_unsorted(startsWith(chs_unsorted,{'ECOG_R'})));
    temp_ECOG_L=sort(chs_unsorted(startsWith(chs_unsorted,{'ECOG_L'})));
    temp_ECOG_LFP = [temp_LFP_R;temp_LFP_L;temp_ECOG_R;temp_ECOG_L];
    chs_final=sort(chs_unsorted(~contains(chs_unsorted,temp_ECOG_LFP)));
    chs_final = [temp_ECOG_LFP;chs_final];
    index_sorted = arrayfun(@(x) find(strcmp(chs_unsorted,x)), chs_final );

    % now sort the other information
    intern_cfg.data.label        = intern_cfg.data.label(index_sorted);
    intern_cfg.data.trial{1}     = intern_cfg.data.trial{1}(index_sorted,:);
    intern_cfg.data.hdr.chanunit = intern_cfg.data.hdr.chanunit(index_sorted);
    intern_cfg.data.hdr.chantype = intern_cfg.data.hdr.chantype(index_sorted);

    intern_cfg.data.hdr.label      = intern_cfg.data.label; % update the other channel names fields

    ii=fieldnames(intern_cfg.channels_tsv);
    for i = 1:numel(ii)
        if ~isempty(intern_cfg.channels_tsv.(ii{i}))
            intern_cfg.channels_tsv.(ii{i}) = intern_cfg.channels_tsv.(ii{i})(index_sorted);
        end
    end


    %% Now assign channel types
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
    % need to make this more generalizable
    %if strcmp(intern_cfg.entities.subject,'EL002') ||  strcmp(intern_cfg.entities.subject,'EL004')
    %     intern_cfg.data.hdr.chanunit   = repmat({'uV'}, intern_cfg.data.hdr.nChans,1);
    %else
    intern_cfg.data.hdr.chanunit   = repmat({'V'}, intern_cfg.data.hdr.nChans,1);




    return;
else
    chs_final  = intern_cfg.channels_tsv.name;
end



%% Initalize containers for BIDS conversion
keySet = {'Rest', 'UPDRSIII', 'SelfpacedRotationL','SelfpacedRotationR',...
    'BlockRotationL','BlockRotationR', 'Evok', 'EvokRamp','SelfpacedSpeech',...
    'ReadRelaxMoveL', 'VigorStimR', 'VigorStimL', 'VigorRailR', 'VigorRailL','SelfpacedHandTapL',...
    'SelfpacedHandTapR', 'SelfpacedHandTapB','Free','DyskinesiaProtocol',...
    'NaturalBehavior', 'StopRailR','StopRailL' ...
    };
descrSet = {'Rest recording', ...
    'Recording performed during part III of the UPDRS (Unified Parkinson''s Disease Rating Scale) questionnaire.',...
    'Selfpaced left wrist rotations performed on custom-built analog rotameter which translates degree of rotation to volt.',...
    'Selfpaced right wrist rotations performed on custom-built analog rotameter which translates degree of rotation to volt.',...
    'Blocks of 30 seconds of rest followed by blocks of 30 seconds of continuous wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt. Performed with the left hand.',...
    'Blocks of 30 seconds of rest followed by blocks of 30 seconds of continuous wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt. Performed with the right hand.',...
    'Evoked potentials recording. Single stimulation pulses of fixed amplitude following a block of a specified frequency.',...
    'Evoked potentials recording. Single stimulation pulses of varying amplitude per block of a specified frequency.',...
    'Selfpaced reading aloud of the fable ''The Parrot and the Cat'' by Aesop. Extended pauses in between sentences.',...
    'Block of 30 seconds of continuous left wrist rotation performed on a custom-built rotameter which translates degree of rotation to volt followed by a block of 30 seconds of rest followed by a block of 30 seconds of reading aloud (''The Parrot and the Cat'' by Aesop). Multiple sets.',...
    'Performance of diagonal forearm movements with a cursor on a screen using a digitizing tablet. Start and stop events are visually cued on screen with a rest duration of 350 ms. 4 blocks with 96 movements each. In blocks 1 an 3 subthalamic deep brain stimulation is applied for 300 ms if a movement is slower/faster than the previous two movements. The order of slow/fast blocks is alternated between participants. Performed with the dominant hand. (right)',...
    'Performance of diagonal forearm movements with a cursor on a screen using a digitizing tablet. Start and stop events are visually cued on screen with a rest duration of 350 ms. 4 blocks with 96 movements each. In blocks 1 an 3 subthalamic deep brain stimulation is applied for 300 ms if a movement is slower/faster than the previous two movements. The order of slow/fast blocks is alternated between participants. Performed with the dominant hand. (left)',...
    'Performance of arm movements using the handle of a custom-made rail. The paradigm is structured into 4 blocks (Mov, Res, Con, Non) in pseudorandomized order. During each block the patient performs 12 movement episodes lasting 4 seconds. Each movement is followed by a break episode of the same length. During the movement episodes the patient is asked to move the handle from one side of the rail to the other as fast as possible. Movement and break episodes are cued on screen. 2 minutes break are forced between blocks. Stimulation during each block: Mov=Stimulation during movement segments (3-10), Res=Stimulation during break segments (3-10), Con=Stimulation during movement and break (3-10), Non=No stimulation.',...
    'Performance of arm movements using the handle of a custom-made rail. The paradigm is structured into 4 blocks (Mov, Res, Con, Non) in pseudorandomized order. During each block the patient performs 12 movement episodes lasting 4 seconds. Each movement is followed by a break episode of the same length. During the movement episodes the patient is asked to move the handle from one side of the rail to the other as fast as possible. Movement and break episodes are cued on screen. 2 minutes break are forced between blocks. Stimulation during each block: Mov=Stimulation during movement segments (3-10), Res=Stimulation during break segments (3-10), Con=Stimulation during movement and break (3-10), Non=No stimulation.',...
    'Selfpaced left hand tapping, circa every 10 seconds, without counting, in resting seated position.',...
    'Selfpaced right hand tapping, circa every 10 seconds, without counting, in resting seated position.',...
    'Bilateral selfpaced hand tapping in rested seated position, one tap every 10 seconds, the patient should not count the seconds. The hand should be raised while the wrist stays mounted on the leg. Correct the pacing of the taps when the tap-intervals are below 8 seconds, or above 12 seconds. Start with contralateral side compared to ECoG implantation-hemisfere. The investigator counts the number of taps and instructs the patients to switch tapping-side after 30 taps, for another 30 taps in the second side.',...
    'Free period, no instructions, this period is recorded during the Dyskinesia-Protocol to monitor the increasing Dopamine-Level',...
    'Total concatenated recording of the dyskinesia protocol, as defined in the lab book',...
    'Natural behavior observed when walking, chatting, drinking or eating at the hospital venue, outside of the experimental lab',...
    'Performance of self-paced arm movements using the handle of a custom-made rail with the right hand. Stop instructions presented visually after decoding of movement intention.', ...
    'Performance of self-paced arm movements using the handle of a custom-made rail with the left hand. Stop instructions presented visually after decoding of movement intention.'...
    };
instructionSet = {'Do not move or speak and keep your eyes open.',...
    'See UPDRS questionnaire.',...
    'Perform 50 wrist rotations with your left hand with an interval of about 10 seconds. Do not count in between rotations.',...
    'Perform 50 wrist rotations with your right hand with an interval of about 10 seconds. Do not count in between rotations.',...
    'Upon the auditory command "start", perform continuous wrist rotations with your left hand, until you perceive the auditory command "stop". Perform these wrist rotations as fast as possible and with the largest possible amplitude.',...
    'Upon the auditory command "start", perform continuous wrist rotations with your right hand, until you perceive the auditory command "stop". Perform these wrist rotations as fast as possible and with the largest possible amplitude.',...
    'Do not move or speak and keep your eyes open.',...
    'Do not move or speak and keep your eyes open.',...
    'Read aloud sentence by sentence the text in front of you. Leave a pause of several seconds in between sentences.',...
    'At the beginning of each block, a text will appear on the screen, specifying the task to be performed. An auditory cue will then be issued, marking the begin of your task. Perform the task until the next cue marks the end of the task. Tasks are either continuous left wrist rotation, resting with open eyes or reading aloud the text displayed on the screen.',...
    'Your task is to move your pen from one side of the screen to the other. When you see a square, please move your pen there and stay on the square until a new square appears on the other side. Then move the pen to the new square. Please move as fast as you can and touch the screen with the pen throughout the whole experiment.',...
    'Your task is to move your pen from one side of the screen to the other. When you see a square, please move your pen there and stay on the square until a new square appears on the other side. Then move the pen to the new square. Please move as fast as you can and touch the screen with the pen throughout the whole experiment.',...
    'Your task is to move the handle from one side to the other as far and as fast as possible as soon as the Move-cue is given on the screen. As soon as you see the Break-cue you should stop moving the handle. Use right hand.',...
    'Your task is to move the handle from one side to the other as far and as fast as possible as soon as the Move-cue is given on the screen. As soon as you see the Break-cue you should stop moving the handle. Use left hand.',...
    'Keep both hands resting on your legs, and tap with your left hand by raising the hand and fingers of your left hand, without letting the arm be lifted from the leg. Do not count in between rotations.',...
    'Keep both hands resting on your legs, and tap with your right hand by raising the hand and fingers of your right hand, without letting the arm be lifted from the leg. Do not count in between rotations.',...
    'Keep both hands resting on your legs. First tap with your left hand (if ECoG is implanted in the right hemisphere; if ECoG is implanted in left hemisphere, start with right hand) by raising the left hand and fingers while the wrist is mounted on the leg. Make one tap every +/- ten seconds. Do not count in between taps. After 30 taps, the recording investigator will instruct you to tap on with your right (i.e. left) hand. After 30 taps the recording investigator will instruct you to stop tapping.',...
    'Free period, without instructions or restrictions, between Rest-measurement and Task-measurements',...
    'Instructions for the dyskinesia protocol, as defined in the lab book',...
    'Free period to be spend at the hospital venue (corridor, canteen, ...), without further instructions',...
    'In this examination, you will perform movements with the handle in front of you. The handle is in the starting position when it is all the way to the left or all the way to the right. In the first phase, you will perform around 40 movements. Perform a single movement in each round. At the beginning of each round, return the handle to the starting position. When you have returned the handle to the starting position, a black cross will appear. Then stand ready for a movement. As soon as the cross appears, you must wait at least 1 second to perform the movement. However, you have up to 10 seconds. You decide when to make the move between 1 and 10 seconds after the cross appears. However, try NOT to count in your head. NOTE: DO NOT move the handle immediately after the cross appears. Wait at least 1 second and then decide for yourself when you want to move. From time to time, the word STOP will randomly appear in red as you move. Then you must react as quickly as possible and try to stop the movement. If you do not manage to stop the movement, it doesnot matter. Simply return the handle to the starting position for the next round.',...
    'In this examination, you will perform movements with the handle in front of you. The handle is in the starting position when it is all the way to the left or all the way to the right. In the first phase, you will perform around 40 movements. Perform a single movement in each round. At the beginning of each round, return the handle to the starting position. When you have returned the handle to the starting position, a black cross will appear. Then stand ready for a movement. As soon as the cross appears, you must wait at least 1 second to perform the movement. However, you have up to 10 seconds. You decide when to make the move between 1 and 10 seconds after the cross appears. However, try NOT to count in your head. NOTE: DO NOT move the handle immediately after the cross appears. Wait at least 1 second and then decide for yourself when you want to move. From time to time, the word STOP will randomly appear in red as you move. Then you must react as quickly as possible and try to stop the movement. If you do not manage to stop the movement, it doesnot matter. Simply return the handle to the starting position for the next round.',...

};
task_descr = containers.Map(keySet,descrSet);
task_instr = containers.Map(keySet,instructionSet);
%task_descr = intern_cfg.task_description;
%task_instr = intern_cfg.task_instructions;

%% Now write data to BIDS

% Initialize a cell array of 'n/a' for practicality
n_a = repmat({'n/a'},intern_cfg.data.hdr.nChans,1);

% adept for each different recording
cfg = [];
cfg.method                  = 'convert';
cfg.bidsroot                = intern_cfg.rawdata_root;
cfg.datatype                = 'ieeg';
cfg.sub                     = intern_cfg.entities.subject;
if ~endsWith( intern_cfg.entities.session , digitsPattern(2))
    error('session does not end on two digits')
elseif xor((startsWith( intern_cfg.entities.session , 'EcogLfpMed')) , (ECOG_contacts>0))
    error('session does not match with ECOG')
elseif xor((startsWith( intern_cfg.entities.session , 'LfpMed')) , (ECOG_contacts==0))
    error('session does not match with LFP only')
else
    cfg.ses                     = intern_cfg.entities.session;
end


%% sort out later why this not work
%     if isfield(intern_cfg,'sessions_tsv')
%
%        %cfg.sessions.acq_date = char(intern_cfg.sessions_tsv.acq_date);
%
%        if ~strcmp(intern_cfg.scans_tsv.acq_time,'n/a')
%             cfg.sessions.acq_date =  char([intern_cfg.scans_tsv.acq_time(1:10)]);%for the sessions.tsv file
%
%        else
%            error('Datetime cannot be missing because the format got updated to YYYY-MM-DDT00:00:00 and date is always known')
%        end
%     else
%         error('session date could not be determined')
%     end

%     if isfield(intern_cfg,'sessions_tsv')
%
%        cfg.sessions.acq_date = char(intern_cfg.sessions_tsv.acq_date);
%
%     else
%        if ~strcmp(intern_cfg.scans_tsv.acq_time,'n/a')
%             cfg.sessions.acq_date =  char([intern_cfg.scans_tsv.acq_time(1:10)]);%for the sessions.tsv file
%             %cfg.sessions.acq_date
%        end
%
%     end

%% Provide info for the scans.tsv file
% the acquisition time can normally be found in the folder name of the recording
cfg.scans.acq_time              =  intern_cfg.scans_tsv.acq_time;
pat1=pattern(digitsPattern(4) + "-" + digitsPattern(2) + "-" + digitsPattern(2) + "T" + digitsPattern(2) + ":" + digitsPattern(2) + ":"+ digitsPattern(2));
pat2=pattern(digitsPattern(4) + "-" + digitsPattern(2) + "-" + digitsPattern(2));
pat3=pattern(digitsPattern(8) + "T" + digitsPattern(6));
%pat4=pattern("20" + digitsPattern(6) + "_" digitsPattern(6));
if matches(intern_cfg.scans_tsv.acq_time,pat1)
    cfg.scans.acq_time = intern_cfg.scans_tsv.acq_time; %known time point
elseif contains(intern_cfg.scans_tsv.acq_time,'n/a')
    error('no datetime found in scans tsv, it contains n/a')
elseif matches(intern_cfg.sessions_tsv.acq_date,pat2)
    cfg.scans.acq_time = [intern_cfg.sessions_tsv.acq_date , 'T00:00:00']; %unknown time point
elseif ~isempty(extract(intern_cfg.inputdata_fname,pat3))
    time_date=extract(intern_cfg.inputdata_fname,pat3);
    time_date=time_date{1};
    cfg.scans.acq_time = strcat(time_date(1:4) , '-' , time_date(5:6) , '-' , time_date(7:11) , ':' , time_date(12:13) , ':' , time_date(14:15));
    intern_cfg.scans_tsv.acq_time = cfg.scans.acq_time; %reverse writing from cfg to intern_cfg
else
    error('no datetime found in sessions tsv neither match in scans.tsv or inputfilename')
end


%%% NEED TO BE FIXED FOR NEW RECORDINGS 17 7 2023
%% sessions.tsv
cfg.sessions.acq_date_no_time = [ char([intern_cfg.scans_tsv.acq_time(1:10)]),  'T00:00:00'];
if contains(cfg.ses, 'OffOn')
    cfg.sessions.medication_state  = 'OFFON';
elseif contains(cfg.ses, 'Off')
    cfg.sessions.medication_state  = 'OFF';
elseif contains(cfg.ses, 'On')
    cfg.sessions.medication_state  = 'ON';
else
    error('medication state could not be derived from session')
end

if ECOG_contacts>0
    UPDRS=readtable('UPDRS_Berlin.xlsx','sheet','recording_detailed_EcogLFP');
else
    UPDRS=readtable('UPDRS_Berlin.xlsx','sheet','recording_detailed_LFP');
end
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

    if isequal(cfg.sessions.acq_date_no_time,'2017-10-18T00:00:00')
        cfg.sessions.subscore_bradykinesia_total = 0; % hack to overcome that this session EL001 ses-EcogLfpMedOff01 is overwritten
    elseif isequal(cfg.sessions.acq_date_no_time,'2022-04-08T00:00:00')
        cfg.sessions.subscore_bradykinesia_total = 0; % hack to overcome that this session L005 ses-EcogLfpOff01 is overwritten
    elseif isequal(cfg.sessions.acq_date_no_time,'2022-04-10T00:00:00')
        cfg.sessions.subscore_bradykinesia_total = 0; % hack to overcome that this session L005 ses-EcogLfpOff02 is overwritten
    elseif isequal(cfg.sessions.acq_date_no_time,'2022-04-25T00:00:00')
        cfg.sessions.subscore_bradykinesia_total = 0; % hack to overcome that this session L006 ses-EcogLfpOff01 is overwritten
    end
end


cfg.task                    = intern_cfg.entities.task;
%cfg.description            is determined at place of stimsettings
cfg.acq                     = intern_cfg.entities.acquisition; %'StimOff';  % add here 'Dopa00' during dyskinesia-protocol recording: e.g. 'StimOffDopa30'.
%%% adapt the cfg.acq acquistion for Evok and EvokRamp later
% cfg.acq in case of StimOn is being set at the end
if isa(intern_cfg.entities.run,'double')
    cfg.run                 = intern_cfg.entities.run;
else
    cfg.run                 = str2num(intern_cfg.entities.run);
end
cfg.space                   = intern_cfg.entities.space; %'MNI152NLin2009bAsym';



% Specify some general information
cfg.InstitutionName                         = 'Charite - Universitaetsmedizin Berlin, corporate member of Freie Universitaet Berlin and Humboldt-Universitaet zu Berlin, Department of Neurology with Experimental Neurology/BNIC, Movement Disorders and Neuromodulation Unit';
cfg.InstitutionAddress                      = 'Chariteplatz 1, 10117 Berlin, Germany';
cfg.dataset_description.Name                = 'BIDS_01_Berlin_Neurophys';
cfg.dataset_description.BIDSVersion         = '1.8.0';
cfg.dataset_description.License             = 'n/a';
cfg.dataset_description.Funding             = {'Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - Project-ID 424778381 - TRR 295'};
cfg.dataset_description.Authors             = {'Jean Beyer', 'Thomas Binns', 'Johannes Busch', 'Alessia Cavallo', 'Meera Chikermane', 'Katharina Faust', 'Lucia Feldmann', 'Jeroen Habets', 'Richard Koehler', 'Andrea Kuehn', 'Roxanne Lofredi', 'Timon Merk', 'Wolf-Julian Neumann', 'Gerd-Helge Schneider', 'Annca Timpert', 'Ulrike Uhlig', 'Jojo Vanhoecke'};
cfg.dataset_description.Acknowledgements    = 'Special thanks to all other people involved in acquiring the data.';


% Provide the long description of the task and participant instructions
cfg.TaskName                = intern_cfg.entities.task; %intern_cfg.ieeg.TaskName;
if isfield(intern_cfg.ieeg,'TaskDescription')
    if ~strcmp(intern_cfg.ieeg.TaskDescription,'')
        cfg.TaskDescription         = intern_cfg.ieeg.TaskDescription;
    else
        cfg.TaskDescription         = task_descr(intern_cfg.entities.task);
    end
else
    cfg.TaskDescription         = task_descr(intern_cfg.entities.task);
end
if isfield(intern_cfg.ieeg,'Instructions')
    if ~strcmp(intern_cfg.ieeg.Instructions,'')
        cfg.Instructions            = intern_cfg.ieeg.Instructions ;
    else
        cfg.Instructions            = task_instr(intern_cfg.entities.task);
    end
else
    cfg.Instructions            = task_instr(intern_cfg.entities.task);
end

% Provide info about recording hardware
if isfield(intern_cfg.ieeg,'Manufacturer')
    hardware_manufacturer = intern_cfg.ieeg.Manufacturer;
else
    error('Please define a valid hardware manufacturer')
end

% NEED HELP FROM RICHARD here
if logical(contains(hardware_manufacturer,'TMSi'))
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
    Hardware_Filters.AnalogueBandwidth = 800;
    cfg.ieeg.SoftwareFilters        = 'no additional filters'; %MUST
    cfg.ieeg.HardwareFilters        = Hardware_Filters; %Recommended
elseif strcmp(hardware_manufacturer,'Alpha Omega Engineering Ltd. (AO)')
    cfg.Manufacturer                = 'Alpha Omega Engineering Ltd. (AO)';
    cfg.ManufacturersModelName      = 'Neuro Omega';
    cfg.SoftwareVersions            = 'n/a';
    cfg.DeviceSerialNumber          = 'n/a';
    %10000 Hz high -> 0.07Hz low
    cfg.channels.low_cutoff         = repmat({'0'},intern_cfg.data.hdr.nChans,1);
    cfg.channels.high_cutoff        = repmat({'n/a'},intern_cfg.data.hdr.nChans,1);
    cfg.ieeg.SoftwareFilters        = 'n/a'; %MUST
    cfg.ieeg.HardwareFilters        = 'n/a'; %Recommended

elseif strcmp(hardware_manufacturer,'Newronika')
    cfg.Manufacturer                = 'Newronika';
    cfg.ManufacturersModelName      = 'n/a';
    cfg.SoftwareVersions            = 'n/a';
    cfg.DeviceSerialNumber          = 'n/a';
    cfg.channels.low_cutoff         = repmat({'0'},intern_cfg.data.hdr.nChans,1);
    cfg.channels.high_cutoff        = repmat({'n/a'},intern_cfg.data.hdr.nChans,1);
    cfg.ieeg.SoftwareFilters        = 'n/a'; %MUST
    cfg.ieeg.HardwareFilters        = 'n/a'; %Recommended
elseif strcmp(hardware_manufacturer,'Brain Products GmbH')
    cfg.Manufacturer                = 'Brain Products GmbH';
    cfg.ManufacturersModelName      = 'n/a';
    cfg.SoftwareVersions            = 'n/a';
    cfg.DeviceSerialNumber          = 'n/a';
    cfg.channels.low_cutoff         = repmat({'0'},intern_cfg.data.hdr.nChans,1);
    cfg.channels.high_cutoff        = repmat({'n/a'},intern_cfg.data.hdr.nChans,1);
    cfg.ieeg.SoftwareFilters        = 'n/a'; %MUST
    cfg.ieeg.HardwareFilters        = 'n/a'; %Recommended
elseif strcmp(hardware_manufacturer,'Cambridge Electronic Design (CED)')
    cfg.Manufacturer                = 'Cambridge Electronic Design (CED)';
    cfg.ManufacturersModelName      = 'n/a';
    cfg.SoftwareVersions            = 'n/a';
    cfg.DeviceSerialNumber          = 'n/a';
    cfg.channels.low_cutoff         = repmat({'0'},intern_cfg.data.hdr.nChans,1);
    cfg.channels.high_cutoff        = repmat({'n/a'},intern_cfg.data.hdr.nChans,1);
    cfg.ieeg.SoftwareFilters        = 'n/a'; %MUST
    cfg.ieeg.HardwareFilters        = 'n/a'; %Recommended

else
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
    Hardware_Filters.AnalogueBandwidth = 800;
    cfg.ieeg.SoftwareFilters        = 'no additional filters'; %MUST
    cfg.ieeg.HardwareFilters        = Hardware_Filters; %Recommended
    %error('Please define a valid hardware manufacturer')
end

% need to check in the LFP excel sheet on the S-drive
% Provide info about the participant
cfg.participants.sex                    = intern_cfg.participants.sex; %found in the clinical data (you probably don't have have access to the SAP, sometimes it can be found in AG-Bewegungsstörungen/Filme)
cfg.participants.handedness             = intern_cfg.participants.handedness; %LFP excel sheet
cfg.participants.age                    = intern_cfg.participants.age ; %LFP excel sheet
cfg.participants.date_of_implantation   = intern_cfg.participants.date_of_implantation ;  %'2022-01-20T00:00:00'; %LFP excel sheet
cfg.participants.UPDRS_III_preop_OFF    = 'n/a';
cfg.participants.UPDRS_III_preop_ON     = 'n/a';
cfg.participants.disease_duration       = 'n/a';
if any(intern_cfg.participants.disease_duration) > 0
    cfg.participants.disease_duration       = intern_cfg.participants.disease_duration;
end
cfg.participants.PD_subtype             = intern_cfg.participants.PD_subtype; %'akinetic-rigid'; %SAP
cfg.participants.symptom_dominant_side  = intern_cfg.participants.symptom_dominant_side;% 'right'; %LFP excel sheet
cfg.participants.LEDD                   = intern_cfg.participants.LEDD; %1600; %calculated from lab book with https://www.parkinsonsmeasurement.org/toolBox/levodopaEquivalentDose.htm
cfg.participants.DBS_target                 = DBS_target;
cfg.participants.DBS_hemisphere             = DBS_hemisphere;
cfg.participants.DBS_manufacturer           = DBS_manufacturer;
cfg.participants.DBS_model                  = DBS_model;
cfg.participants.DBS_directional            = DBS_directional;
cfg.participants.DBS_contacts               = DBS_contacts;
cfg.participants.DBS_description            = DBS_description;

% Info about the ECOG electrode
cfg.participants.ECOG_target                = ECOG_target_long;
cfg.participants.ECOG_hemisphere            = ECOG_hemisphere;
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
if isfield(intern_cfg, 'electrodes_tsv')
    if ~isempty(intern_cfg.electrodes_tsv.name)
        no_need_for_new_electrodes_tsv = true;
    else
        no_need_for_new_electrodes_tsv = false;
    end
else
    no_need_for_new_electrodes_tsv = false;
end

if no_need_for_new_electrodes_tsv

    cfg.electrodes.name         = intern_cfg.electrodes_tsv.name;
    sens.label                  = intern_cfg.electrodes_tsv.name;
    sens.elecpos                = str2double([intern_cfg.electrodes_tsv.x,intern_cfg.electrodes_tsv.y,intern_cfg.electrodes_tsv.z]);
    cfg.elec                    = sens;
    %cfg.electrodes.size        = intern_cfg.electrodes_tsv.size; -> need to create new when empty
    cfg.electrodes.material     = intern_cfg.electrodes_tsv.material;
    cfg.electrodes.manufacturer = intern_cfg.electrodes_tsv.manufacturer;
    %cfg.electrodes.group       -> need to overwrite
    cfg.electrodes.hemisphere   = intern_cfg.electrodes_tsv.hemisphere;
    cfg.electrodes.type         = intern_cfg.electrodes_tsv.type;
    cfg.electrodes.impedance    = intern_cfg.electrodes_tsv.impedance;
    cfg.electrodes.dimension    = intern_cfg.electrodes_tsv.dimension;

    cfg.electrodes.group        =  repmat({'n/a'},length(cfg.electrodes.name),1);

    cfg.electrodes.group(startsWith(cfg.electrodes.name, 'ECOG_R')) = {'ECOG_right'};
    cfg.electrodes.group(startsWith(cfg.electrodes.name, 'ECOG_L')) = {'ECOG_left'};
    cfg.electrodes.group(startsWith(cfg.electrodes.name, 'LFP_R')) = {'DBS_right'};
    cfg.electrodes.group(startsWith(cfg.electrodes.name, 'LFP_L')) = {'DBS_left'};


    %check for the side of the ecog strip
    %first column is x > +1
    assert(all(sens.elecpos((startsWith(sens.label, 'ECOG_R')),1) >= 0))
    %first column is x < -1
    assert(all(sens.elecpos((startsWith(sens.label, 'ECOG_L')),1) <= 0))


else

    % REQUIRED. Name of the electrode
    % need to build if for loop => what to do if the coordsys file already
    % there is.
    % here: on this place would the electrode localization file


    % extract the channel names that contain LFP or ECOG
    %sens.label = chs_final(contains(string(chs_final),["LFP", "ECOG"]));

    % create extra channels in case some channels are combined together
    sens.label = {};
    if any(contains(string(chs_final),["ECOG_R"]))
        for i=1:ECOG_contacts
            sens.label{end+1} = strcat('ECOG_R_', sprintf( '%02d', i ), '_', ECOG_target, '_', char(ECOG_manufacturer_short));
        end
    end
    if any(contains(string(chs_final),["ECOG_L"]))
        for i=1:ECOG_contacts
            sens.label{end+1} = strcat('ECOG_L_', sprintf( '%02d', i ), '_', ECOG_target, '_', char(ECOG_manufacturer_short));
        end
    end
    if any(contains(string(chs_final),["LFP_R"]))
        for i=1:DBS_contacts
            sens.label{end+1} = strcat('LFP_R_', sprintf( '%02d', i ), '_', DBS_target, '_', char(DBS_manufacturer_short));
        end
    end
    if any(contains(string(chs_final),["LFP_L"]))
        for i=1:DBS_contacts
            sens.label{end+1} = strcat('LFP_L_', sprintf( '%02d', i ), '_', DBS_target, '_', char(DBS_manufacturer_short));
        end
    end

    sens.label =sens.label';

    % Electrode positions are imported from external files (e.g. Lead-DBS
    % ea_reconstruction.mat) and a sens FieldTrip struct is created
    % (see FT_DATATYPE_SENS)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % To Do: -> for now fixed, if else statement fixed for models: Cartesia X and Sensight
    % this is the channel position for medtronic
    sens.chanpos = [
        zeros(ECOG_contacts, 3);...
        zeros(DBS_contacts, 3); ...
        zeros(DBS_contacts, 3)
        ];
    % this is for medtronic
    if isfield(intern_cfg,'ECOG_localization')
        error('what is this function doing?') %need to perhaps swap the order of ECOG with DBS
        sens.chanpos(DBS_contacts+DBS_contacts + 1 : end,1:3) = intern_cfg.ECOG_localization;
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
        repmat({ECOG_material}, ECOG_contacts ,1);
        repmat({DBS_material},DBS_contacts*2,1)
        ];
    cfg.electrodes.manufacturer = [
        repmat({cfg.participants.ECOG_manufacturer},ECOG_contacts,1);
        repmat({cfg.participants.DBS_manufacturer},DBS_contacts*2,1)
        ];
    cfg.electrodes.group        = [
        repmat({['ECOG_' cfg.participants.ECOG_hemisphere]},ECOG_contacts,1);
        repmat({'DBS_right'},DBS_contacts,1);
        repmat({'DBS_left'},DBS_contacts,1)
        ];
    cfg.electrodes.hemisphere   = [
        repmat({ECOG_hemisphere},ECOG_contacts,1);
        repmat({'right'},DBS_contacts,1);
        repmat({'left'},DBS_contacts,1)
        ];
    % RECOMMENDED. Type of the electrode (e.g., cup, ring, clip-on, wire, needle)
    cfg.electrodes.type         = [
        repmat({'strip'},ECOG_contacts,1);%=> this is the ECOG
        repmat({'depth'},DBS_contacts*2,1) %=> this is DBS
        ];

    % RECOMMENDED. Impedance of the electrode in kOhm
    cfg.electrodes.impedance    = repmat({'n/a'},length(sens.label),1);
    cfg.electrodes.dimension    = [
        repmat({sprintf('[1x%d]',ECOG_contacts)},ECOG_contacts,1);
        repmat({sprintf('[1x%d]',DBS_contacts)},DBS_contacts*2,1)
        ];
end

% TO DO: need to add the ECOG contact size at the end and make it 6 or
% 12 long
% define size of single electrode contacts
% if 1 DBS contact per level: size=6; if 3 contacts per level: s=1.5
add_ECOG_size = 1;


if contains(cfg.participants.DBS_model,'SenSight')
    cfg.electrodes.size = {
        6 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
        6 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
        }; %  ECoG contacts std 4.15
elseif strcmp(cfg.participants.DBS_model,'Vercise Cartesia') %= 'Vercise Directed'
    cfg.electrodes.size = {
        6 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
        6 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
        }; %  ECoG contacts std 4.15
elseif strcmp(cfg.participants.DBS_model,'Vercise Standard')
    cfg.electrodes.size = {
        6 6 6 6 6 6 6 6 ...
        6 6 6 6 6 6 6 6 ...
        }; %  ECoG contacts std 4.15
elseif strcmp(cfg.participants.DBS_model, 'Vercise Cartesia X')
    cfg.electrodes.size = {
        1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
        1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 6 ...
        }; %  ECoG contacts std 4.15
elseif (strcmp(cfg.participants.DBS_model, 'Model 3387') || strcmp(cfg.participants.DBS_model, 'Model 3389')) % Medtronic
    cfg.electrodes.size = {
        6 6 6 6 ...
        6 6 6 6 ...
        }; %  ECoG contacts std 4.15
end
% in e.g. subject L005, there are segmented leads, with aberrant electrode sizes
% in e.g. subject L007, there are brain facing and skull facing contacts
if isfield(intern_cfg.electrodes_tsv,'size')
    if ~isempty(intern_cfg.electrodes_tsv.size)
        cfg.electrodes.size = intern_cfg.electrodes_tsv.size;
        add_ECOG_size = 0;
    else
        if isempty(cfg.electrodes.size)
            error('no electrode size')
        end
    end
else
    error('no electrode size')
end

if add_ECOG_size
    if (ECOG_contacts == 6) &&  strcmp(ECOG_model,'TS06R-AP10X-0W6')
        cfg.electrodes.size(1:end+6) = [{2.54} {2.54} {2.54} {2.54} {2.54} {2.54} cfg.electrodes.size(:)' ];
    elseif (ECOG_contacts == 12) && strcmp(ECOG_model,'DS12A-SP10X-000')
        cfg.electrodes.size(1:end+12) = [{4.15} {4.15} {4.15} {4.15} {4.15} {4.15} {4.15} {4.15} {4.15} {4.15} {4.15} {4.15} cfg.electrodes.size(:)'];
    elseif (ECOG_contacts == 0) && strcmp(ECOG_model,'n/a')
        %continue
    else
        error('no ECOG size specified')
    end
end



%% Provide special channel info
if isfield(intern_cfg,'channels_tsv')
    if ~isfield(intern_cfg.channels_tsv,'notch')
        intern_cfg.channels_tsv.notch = n_a;
    end
end
if isfield(intern_cfg,'channels_tsv')
    if ~isempty(intern_cfg.channels_tsv.name) && ...
            ~isempty(intern_cfg.channels_tsv.type) && ...
            ~isempty(intern_cfg.channels_tsv.units) && ...
            ~isempty(intern_cfg.channels_tsv.low_cutoff) && ...
            ~isempty(intern_cfg.channels_tsv.high_cutoff) && ...
            ~isempty(intern_cfg.channels_tsv.reference) && ...
            ~isempty(intern_cfg.channels_tsv.group) && ...
            ~isempty(intern_cfg.channels_tsv.sampling_frequency) && ...
            ~isempty(intern_cfg.channels_tsv.notch) && ...
            ~isempty(intern_cfg.channels_tsv.status) && ...
            ~isempty(intern_cfg.channels_tsv.status_description)

        %         cfg.channels.status             = intern_cfg.channels_tsv.status;
        %         cfg.channels.status_description = intern_cfg.channels_tsv.status_description;
        %         cfg.channels.name               = intern_cfg.channels_tsv.name;
        %         cfg.channels.type               = intern_cfg.channels_tsv.type;
        cfg.channels = intern_cfg.channels_tsv; %these are recording-specific, and often manually updated or maintained
    else
        if ECOG_contacts == 12
            error('need to fix code so that number channels.tsv can have fewer ECOG contacts based on the model, because sometimes they are empty or deleted')
        end
        %from the python input file


        %eeg_emg_channel = find(startsWith(intern_cfg.data.label,'EEG') + startsWith(intern_cfg.data.label,'EMG'));

        for chan= 1:length(intern_cfg.channels_tsv.name)
            if ~any(intern_cfg.data.trial{1}(chan,1:10000))
                intern_cfg.channels_tsv.status{chan}='bad';
                if strcmp(intern_cfg.channels_tsv.status_description{chan},'n/a')
                    intern_cfg.channels_tsv.status_description{chan}='Empty';
                end
            end
        end
        cfg.channels.status             = intern_cfg.channels_tsv.status;
        cfg.channels.status_description = intern_cfg.channels_tsv.status_description;

        % settings that are mostly applicable, but now cut out
        cfg.channels.name               = chs_final;
        cfg.channels.type               = intern_cfg.data.hdr.chantype;

        % MOSTLY Always reset the channels references
        typeSet = {'EEG', 'ECOG', 'DBS', 'SEEG', 'EMG', 'ECG', 'MISC'};
        cfg.ieeg.iEEGReference = intern_cfg.ieeg.iEEGReference ;
        refSet = {cfg.ieeg.iEEGReference, cfg.ieeg.iEEGReference, cfg.ieeg.iEEGReference, cfg.ieeg.iEEGReference, 'bipolar', 'bipolar', 'n/a'};
        ref_map = containers.Map(typeSet,refSet);
        cfg.channels.reference = arrayfun(@(ch_type) {ref_map(ch_type{1})}, cfg.channels.type);
        cfg.channels.reference(startsWith(cfg.channels.name,{'EMG_L_R','EMG_R_R'})) = {cfg.ieeg.iEEGReference};
        cfg.channels.status(find(contains(cfg.channels.name, cfg.ieeg.iEEGReference)))={'bad'}
        cfg.channels.status_description(find(contains(cfg.channels.name, cfg.ieeg.iEEGReference)))={'Reference electrode'}

        %MOSTLY always notch filter on n/a
        cfg.channels.notch              = n_a;
        cfg.channels.units              = intern_cfg.data.hdr.chanunit;

        sf = cell(length(chs_final),1);
        sf(:) = {intern_cfg.data.fsample};
        cfg.channels.sampling_frequency = sf;

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

    end
end
% Reference to the iEEG

if isfield(intern_cfg.ieeg,'iEEGReference')
    if ~strcmp(intern_cfg.ieeg.iEEGReference,'n/a') && ~strcmp(intern_cfg.ieeg.iEEGReference,'')
        cfg.ieeg.iEEGReference = intern_cfg.ieeg.iEEGReference;
    end
end


% these are iEEG specific
cfg.ieeg.PowerLineFrequency     = 50;   % since recorded in the Europe
cfg.ieeg.iEEGGround             = 'Right shoulder patch';
hdr = ft_read_header(intern_cfg.inputdata_location, 'checkmaxfilter', false, 'readbids', false);
cfg.ieeg.RecordingDuration               = (hdr.nTrials*hdr.nSamples)/hdr.Fs; %to prevent error for dummy data
% this is to be specified for each model
format_groups = '%s subdural cortical strip and %s %s deep brain stimulation (DBS) leads.';

%cfg.ieeg.iEEGPlacementScheme    = 'Left subdural cortical strip and bilateral subthalamic nucleus (STN) deep brain stimulation (DBS) leads.';
cfg.ieeg.iEEGPlacementScheme =sprintf(format_groups,cfg.participants.ECOG_hemisphere, cfg.participants.DBS_hemisphere, intern_cfg.participants.DBS_target);
format_groups = 'ECOG_%s: %d-contact, 1x%d dual-sided long-term monitoring %s strip on %s. DBS_left: 1x%d %s %s DBS lead in left %s, DBS_right: 1x%d %s %s DBS lead in right %s.';
cfg.ieeg.iEEGElectrodeGroups = sprintf(format_groups, cfg.participants.ECOG_hemisphere, ECOG_contacts, ECOG_contacts, ECOG_manufacturer, ECOG_target_long,...
    DBS_contacts,  DBS_manufacturer, DBS_model, DBS_target,...
    DBS_contacts,  DBS_manufacturer, DBS_model, DBS_target );
% cfg.ieeg.iEEGElectrodeGroups    = 'ECOG_strip: 6-contact, 1x6 dual sided long term monitoring AdTech strip on left sensorimotor cortex, DBS_left: 1x16 Boston Scientific directional DBS lead (Cartesia X) in left STN, DBS_right: 1x16 Boston Scientific directional DBS lead (Cartesia X) in right STN.';


cfg.ieeg.RecordingType          = 'continuous';
if contains(cfg.acq, ["StimOn", "StimOffOn",  "Burst"])
    cfg.ieeg.ElectricalStimulation  = true;
else
    cfg.ieeg.ElectricalStimulation  = false;
end


if cfg.ieeg.ElectricalStimulation
    if contains(cfg.task, 'Evok')
        exp.StimulationMode           = "time-varying";
        exp.StimulationParadigm       = "single pulse stimulation in block-paradigm";
    else
        exp.StimulationMode           = "continuous";
        exp.StimulationParadigm       = "continuous stimulation";
    end
    if startsWith(intern_cfg.entities.acquisition,'EStim')
        exp.StimulationTarget     = ECOG_target_long;
        if isfield(intern_cfg.stim, 'L')
            for i=1:length(intern_cfg.stim.L.CathodalContact)
                assert(or(startsWith(intern_cfg.stim.L.CathodalContact{i},'ECOG'),startsWith(intern_cfg.stim.L.AnodalContact{i},'ECOG')))
                if any([isempty(intern_cfg.stim.L.AnodalContact),contains(intern_cfg.stim.L.AnodalContact,"Ground"),contains(intern_cfg.stim.L.CathodalContact,"Ground")])
                    exp.SimulationMontage         = "monopolar";
                else
                    assert(extract_contact_number(intern_cfg.stim.L.CathodalContact{i})<extract_contact_number(intern_cfg.stim.L.AnodalContact{1}))
                end
            end
        else
            for i=1:length(intern_cfg.stim.R.CathodalContact)
                assert(or(startsWith(intern_cfg.stim.R.CathodalContact{i},'ECOG'),startsWith(intern_cfg.stim.R.AnodalContact{i},'ECOG')))
                if any([isempty(intern_cfg.stim.R.AnodalContact),contains(intern_cfg.stim.R.AnodalContact,"Ground"),contains(intern_cfg.stim.R.CathodalContact,"Ground")])
                    exp.SimulationMontage         = "monopolar";
                else
                    exp.SimulationMontage         = "bipolar";
                    assert(extract_contact_number(intern_cfg.stim.R.CathodalContact{i}) ~= extract_contact_number(intern_cfg.stim.R.AnodalContact{1}))
                end
            end
        end
    else
        exp.StimulationTarget         = DBS_target;
        exp.SimulationMontage         = "monopolar";
    end
    % under if cfg.ieeg.ElectricalStimulation
    if contains(cfg.task, 'Vigor')

        if contains(cfg.acq, 'Cont')
            exp.StimulationMode           = "continuous";
            exp.StimulationParadigm       = "continuous stimulation";
        else
            exp.StimulationMode           = "time-varying";
            exp.StimulationParadigm       = "speed-adaptive DBS";
        end
        if contains(cfg.task, 'VigorRail')
            exp.StimulationMode           = "time-varying";
            exp.StimulationParadigm       = "block-paradigm";
        end

        %if isfield(intern_cfg.entities,'description')
        %    cfg.desc = intern_cfg.entities.description;
        %else
        %    cfg.desc = 'neurophys';
        %end
    end

    %%% correct the stim settings from metadata
    if isfield(intern_cfg.ieeg,'ElectricalStimulationParameters')
        cfg.ieeg.ElectricalStimulationParameters = intern_cfg.ieeg.ElectricalStimulationParameters;
        if isstruct(cfg.ieeg.ElectricalStimulationParameters)
            if isfield(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting, 'Right')
                if isfield(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right, 'StimulationStatus')
                    assert(or(strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus,'OFF'),...
                        strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus,'ON')))
                else
                    if strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right,'n/a')
                        cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right = struct();
                        cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus="OFF";
                    elseif strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right,'OFF')
                        cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right = struct();
                        cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus="OFF";
                    elseif strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.InitialPulseShape,'rectangular') %to check whether there was stimulation
                        cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus="ON";
                    end
                end
            end
            if isfield(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting, 'Left')
                if isfield(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right, 'StimulationStatus')
                    assert(or(strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus,'OFF'),...
                        strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus,'ON')))
                else
                    if strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left,'n/a')
                        cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left = struct();
                        cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.StimulationStatus="OFF";
                    elseif strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left,'OFF')
                        cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left = struct();
                        cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.StimulationStatus="OFF";
                    elseif strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.InitialPulseShape,'rectangular') %to check whether there was stimulation
                        cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.StimulationStatus="ON";
                    end
                end
            end
        end
    else %%% file from BIDS input
        % Enter EXPERIMENTAL stimulation settings
        % these need to be written in the lab book

        exp.DateOfSetting             = intern_cfg.stim.DateOfSetting; %"2021-11-11"

        if ~isfield(intern_cfg.stim, 'L')
            %L = 'OFF';
            L.StimulationStatus           = "OFF";
        else
            if strcmpi(intern_cfg.stim.L,"OFF")
                %L = 'OFF';
                L.StimulationStatus           = "OFF";
            else
                L.CathodalContact             = intern_cfg.stim.L.CathodalContact;
                L.CathodalContactDirection      = "none";

                if isfield(intern_cfg.stim.L, 'AnodalContact')
                    if ~isempty(intern_cfg.stim.L.AnodalContact)
                        L.AnodalContact               = intern_cfg.stim.L.AnodalContact;
                    else
                        %error('no anodal contact')
                        L.AnodalContact               = "Ground";
                    end
                else
                    error('no anodal contact')
                    L.AnodalContact               = "Ground";
                end

                L.AnodalContactDirection    = "omni"; %need to double check
                L.AnodalContactImpedance    = "n/a";
                L.StimulationAmplitude        = intern_cfg.stim.L.StimulationAmplitude;
                L.StimulationPulseWidth       = 60;
                L.StimulationFrequency        = intern_cfg.stim.L.StimulationFrequency;
                L.StimulationStatus           = "ON";
                L.InitialPulseShape           = "rectangular";
                L.InitialPulseWidth           = 60;
                L.InitialPulseAmplitude       = -1.0*L.StimulationAmplitude;
                L.InterPulseDelay             = 0;
                L.SecondPulseShape            = "rectangular";
                L.SecondPulseWidth            = 60;
                L.SecondPulseAmplitude        = L.StimulationAmplitude;
                L.PostPulseInterval           = "n/a";

                if strcmp(cfg.task,'EvokRamp')
                    L.StimulationAmplitude = "time-varying";
                    L.InitialPulseAmplitude = "time-varying";
                    L.SecondPulseAmplitude = "time-varying";
                    L.StimulationAmplitudeMin = intern_cfg.stim.L.StimulationAmplitudeMin;
                    L.StimulationAmplitudeMax = intern_cfg.stim.L.StimulationAmplitudeMax;
                    L.StimulationAmplitudeStepsize = intern_cfg.stim.L.StimulationAmplitudeStepsize;
                end
                if contains(cfg.task,'Evok')
                    L.StimulationPulseWidth  = 90;
                    L.InitialPulseWidth = 90;
                    L.SecondPulseWidth = 90;
                    L.InterPulseDelay = 53;
                    assert(~isempty(L.CathodalContact))
                    assert(~isempty(L.AnodalContact))
                    cfg.acq = [cfg.acq abbrev(L.CathodalContact{1},L.AnodalContact{1})];
                end
            end
        end
        exp.Left                      = L;

        if ~isfield(intern_cfg.stim, 'R')
            %R = 'OFF';
            R.StimulationStatus           = "OFF";
        else
            if strcmpi(intern_cfg.stim.R,"OFF")
                %R = 'OFF';
                R.StimulationStatus           = "OFF";
            else
                R.CathodalContact             = intern_cfg.stim.R.CathodalContact;
                R.CathodalContactDirection      = "none";

                if isfield(intern_cfg.stim.R, 'AnodalContact')
                    if ~isempty(intern_cfg.stim.R.AnodalContact)
                        R.AnodalContact               = intern_cfg.stim.R.AnodalContact;
                    else
                        %error('no anodal contact')
                        R.AnodalContact               = "Ground";
                    end
                else
                    error('no anodal contact')
                    R.AnodalContact               = "Ground";
                end
                R.AnodalContactDirection    = "omni";
                R.AnodalContactImpedance    = "n/a";
                R.StimulationAmplitude        = intern_cfg.stim.R.StimulationAmplitude;
                R.StimulationPulseWidth       = 60;
                R.StimulationFrequency        = intern_cfg.stim.R.StimulationFrequency;
                R.StimulationStatus           = "ON";
                R.InitialPulseShape           = "rectangular";
                R.InitialPulseWidth           = 60;
                R.InitialPulseAmplitude       = -1.0*R.StimulationAmplitude;
                R.InterPulseDelay             = 0;
                R.SecondPulseShape            = "rectangular";
                R.SecondPulseWidth            = 60;
                R.SecondPulseAmplitude        = R.StimulationAmplitude;
                R.PostPulseInterval           = "n/a";

                if strcmp(cfg.task,'EvokRamp')
                    R.StimulationAmplitude = "time-varying";
                    R.InitialPulseAmplitude = "time-varying";
                    R.SecondPulseAmplitude = "time-varying";
                    R.StimulationAmplitudeMin = intern_cfg.stim.R.StimulationAmplitudeMin;
                    R.StimulationAmplitudeMax = intern_cfg.stim.R.StimulationAmplitudeMax;
                    R.StimulationAmplitudeStepsize = intern_cfg.stim.R.StimulationAmplitudeStepsize;
                end
                if contains(cfg.task,'Evok')
                    R.StimulationPulseWidth  = 90;
                    R.InitialPulseWidth = 90;
                    R.SecondPulseWidth = 90;
                    R.InterPulseDelay = 53;
                    assert(~isempty(R.CathodalContact))
                    assert(~isempty(R.AnodalContact))
                    cfg.acq = [cfg.acq abbrev(R.CathodalContact{1},R.AnodalContact{1})];
                end
            end
        end
        exp.Right                     = R;

        param.BestClinicalSetting                = "Berlin parameter preset";
        param.CurrentExperimentalSetting         = exp;
        cfg.ieeg.ElectricalStimulationParameters = param;
    end

    % under if cfg.ieeg.ElectricalStimulation
    if contains(cfg.acq, 'StimOn') && isfield(cfg.ieeg.ElectricalStimulationParameters,'CurrentExperimentalSetting')
        if startsWith(cfg.acq, 'StimOnL') || startsWith(cfg.acq, 'EStimOnL')
            assert(strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus,'OFF'))
            assert(strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.StimulationStatus,'ON'))
        elseif startsWith(cfg.acq, 'StimOnR') || startsWith(cfg.acq, 'EStimOnR')
            assert(strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus,'ON'))
            assert(strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.StimulationStatus,'OFF'))
        elseif startsWith(cfg.acq, 'StimOnB') || startsWith(cfg.acq, 'EStimOnB')
            assert(strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus,'ON'))
            assert(strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.StimulationStatus,'ON'))
        elseif strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus,'OFF') && strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.StimulationStatus,'ON')
            cfg.acq = replace(cfg.acq,'StimOn','StimOnL');
        elseif strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus,'ON') && strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.StimulationStatus,'OFF')
            cfg.acq = replace(cfg.acq,'StimOn','StimOnR');
        elseif strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus,'ON') && strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.StimulationStatus,'ON')
            cfg.acq = replace(cfg.acq,'StimOn','StimOnB');
        else
            error('Stim status unknown')
        end
    end

    % under if cfg.ieeg.ElectricalStimulation
    if contains(cfg.acq, 'StimOn')
        if ~(contains(cfg.acq, 'StimOnL') || contains(cfg.acq, 'StimOnR') || contains(cfg.acq, 'StimOnB') || contains(cfg.acq, 'StimOnX'))
            cfg.acq = replace(cfg.acq,'StimOn','StimOnX');
            assert(strcmp(cfg.ieeg.ElectricalStimulationParameters,'n/a') || ischar(cfg.ieeg.ElectricalStimulationParameters))
        end
    elseif contains(cfg.acq, 'StimOffOn')
        if ~(contains(cfg.acq, 'StimOffOnL') || contains(cfg.acq, 'StimOffOnR') || contains(cfg.acq, 'StimOffOnB') || contains(cfg.acq, 'StimOffOnX'))
            cfg.acq = replace(cfg.acq,'StimOffOn','StimOffOnX');
            assert(strcmp(cfg.ieeg.ElectricalStimulationParameters,'n/a') || ischar(cfg.ieeg.ElectricalStimulationParameters))
        end
    end
    % under if cfg.ieeg.ElectricalStimulation
    cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.SimulationMontage         = "monopolar";
    if strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.StimulationStatus,'ON')
        if and(contains(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.AnodalContact, {'ECOG','LFP'}),...
            contains(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Right.CathodalContact, {'ECOG','LFP'}))
            cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.SimulationMontage         = "bipolar";
        end
    elseif strcmp(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.StimulationStatus,'ON')
        if and(contains(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.AnodalContact, {'ECOG','LFP'}),...
            contains(cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.Left.CathodalContact, {'ECOG','LFP'}))
            cfg.ieeg.ElectricalStimulationParameters.CurrentExperimentalSetting.SimulationMontage         = "bipolar";
        end
    end

end

% specify ieeg specific information

DataNotes = readtable('Data_Notes_Berlin.xlsx','sheet','TO_JSON','Range','A:K');
total_name =  ['sub-', cfg.sub, '_ses-', cfg.ses, '_task-', cfg.task, '_acq-',cfg.acq,'_run-',num2str(cfg.run)];
rownr = find((contains(DataNotes.Filename, total_name)));
if size(rownr)==[1,1]
    cfg.ieeg.Recording_notes        = DataNotes.recording_notes{rownr};
    cfg.ieeg.Data_use               = DataNotes.data_use{rownr};
    cfg.ieeg.Reference_description  = DataNotes.reference_description{rownr};
    cfg.ieeg.Subject_notes          = DataNotes.subject_notes{rownr};
end


end

function contact_number = extract_contact_number(channel_name)
contact_number = str2num(channel_name(isstrprop(channel_name, 'digit')));
end

function a = abbrev(cathode, anode)
    a = '';
    if startsWith(cathode, 'ECOG')
        a = [a 'E' num2str(extract_contact_number(cathode))]; 
    elseif startsWith(cathode, 'LFP')
        a = [a 'D' num2str(extract_contact_number(cathode))]; 
    end
    if startsWith(anode, 'ECOG')
        a = [a 'E' num2str(extract_contact_number(anode))]; 
    elseif startsWith(anode, 'LFP')
        a = [a 'D' num2str(extract_contact_number(anode))]; 
    end
    a = strrep(a,'0','');
end