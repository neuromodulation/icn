function D = wjn_import_AO(filename,channels)



d = load(filename);
fields = fieldnames(d);
data =[];

if ~exist('channels','var')
    channels = fields(ci('_KHz_orig',fields)-2);
    channels(ci('CADD_',channels))=[];
    
    an = ci('ANAL',channels);

    for a =1:length(an)
        if numel((unique(d.(channels{an(a)}))))<100
            rm(a)=an(a);
        else
            rm(a)=0;
        end
    end
    rm(rm==0)=[];
    channels(rm)=[];
    
    for a =1:length(channels)
        tmp = strsplit(channels{a},'__')
        try
            if length(tmp{end})>4 && tmp{end}(1:5) == 'CANAL'
                chanlabels{a} = strrep(tmp{end}(2:end),'_IN_','');
            else
                chanlabels{a} = strrep(tmp{end},'_','');
            end
        catch
            chanlabels{a} = channels{a}
        end
        chanlabels{a} = channels{a} % here I just keep all the naming convention
    end
else
    chanlabels=channels;
end

for a = 1:length(channels)
    i=ci(channels{a},fields);
    if a == 1
        fsample = d.(fields{i(2)})*1000;
        timewin = [d.(fields{i(5)}) d.(fields{i(6)})];
    elseif a>1 && d.(fields{i(2)}) == fsample/1000
        data(a,:,1) = d.(fields{i(1)});
    else
        rd=resample(double(d.(fields{i(1)})),fsample,1000*d.(fields{i(2)}));
        data(a,1:length(rd),1) = rd;
    end
end

T = linspace(timewin(1), timewin(2), size(data,2));
t = linspace(0,size(data,2)/fsample,size(data,2));


iSF = ci({'CStim','Ports','Channel','SF'},fields);

for a = 1:length(iSF)
    info.(fields{iSF(a)}) = d.(fields{iSF(a)});
end

BIDS_struct_save = struct;
BIDS_struct_save.data = data;
BIDS_struct_save.labels = chanlabels;
BIDS_struct_save.fsample = fsample;

save(strcat('BIDS_save_', filename), 'BIDS_struct_save')

no_further_processing = 0;

if no_further_processing == 0

    D=wjn_import_rawdata(['spmeeg_' filename],double(data),chanlabels, fsample);
    info.T = T;
    D.AO = info;
    save(D)

    if isfield(D.AO,'CStimMarker_1')
        S = D.AO.SF_STIM_PARAMS;
        D.AO.STIM_ONSET = S(1,2:end)/(D.AO.SF_STIM_PARAMS_KHz*1000)-D.AO.T(1);
        D.AO.STIM_OFFSET = S(10,2:end)/1000+D.AO.STIM_ONSET;
        D.AO.STIM_CHANNEL = D.chanlabels(S(2,2:end)-10015);
        D.AO.STIM_CHANLIST = unique(D.chanlabels(S(2,2:end)-10015));
        D.AO.STIM_AMP = S(5,2:end);
        D.AO.STIM_FREQ = S(11,2:end);
        D.AO.STIM_PULSEWIDTH = S(6);
        D.AO.STIM_EVENT_ON = strrep(strcat({'STIM_ON_'},num2str(D.AO.STIM_FREQ'),{'Hz_'},D.AO.STIM_CHANNEL',{'_'},num2str(D.AO.STIM_AMP'),{'uA'}),' ','');
        D.AO.STIM_EVENT_OFF = strrep(strcat({'STIM_OFF_'},num2str(D.AO.STIM_FREQ'),{'Hz_'},D.AO.STIM_CHANNEL',{'_'},num2str(D.AO.STIM_AMP'),{'uA'}),' ','');

    end

    iecog = ci('ECOG',D.chanlabels);
    if ~isempty(iecog)
        D=wjn_ecog_rereference(D.fullfile);
    end

    % istnl = ci({'STNL8','STNL1'},D.chanlabels)
    % addchan = {};
    % idata = [];
    % if length(istnl) == 2
    %     idata(end+1,:) = D(istnl(2),:)-D(istnl(1),:);
    %     addchan = {'STNL18'};
    % end
    % 
    % istnr = ci({'STNR8','STNR1'},D.chanlabels)
    % if length(istnr) == 2
    %     idata(end+1,:) = D(istnr(2),:)-D(istnr(1),:);
    %     addchan = [addchan {'STNR18'}];
    % end
    % if ~isempty(addchan)
    %     D=wjn_add_channels(D.fullfile,idata,addchan)
    % end
    D=chantype(D,':',wjn_chantype(D.chanlabels));
    save(D);
end

    