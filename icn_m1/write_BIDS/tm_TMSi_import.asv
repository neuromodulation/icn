function D=wjn_TMSi_import(filename,chans)

d = TMSi.Poly5.read(filename);

if ~exist('chans','var')
    for a  = 1:length(d.channels)
        chans{a} = d.channels{a}.name;
    end
end

nfname = strrep(strrep(filename(1:end-6),'.','_'),' ','_');
%keyboard
D=wjn_import_rawdata(nfname,d.samples,chans,d.sample_rate);

D=wjn_remove_channels(D.fullfile,{'X','Y','Z','AUX 2-1','AUX 2-2','AUX 2-3','X-AXIS','Y-AXIS','Z-AXIS', 'STATUS', 'COUNTER', 'Counter 2power24'});
D=tm_remove_bad_time_segments(D.fullfile,[0 1;D.time(end)-1 D.time(end)], "keep",filename);


% THIS IS THE VERSION OF SUB002
%D=wjn_remove_channels(D.fullfile,{'Stat','STATUS','COUNTER','Counter 2power24','SaO2','Pleth','HRate','Status','Saw'});
%D=tm_remove_bad_time_segments(D.fullfile,[0 1;D.time(end)-1 D.time(end)], "keep",filename);


%D=wjn_filter(D.fullfile,2,'high');
%D=wjn_filter(D.fullfile,[48 52],'stop');
%D=wjn_filter(D.fullfile,98,'low');
% 
% D=wjn_tf_wavelet(D.fullfile,1:100,15);