function D=tm_remove_bad_time_segments(filename,segments,mode, f_name_save)

if ~exist('mode','var')
    mode = 'remove';
end
D=wjn_sl(filename);

if D.ntrials > 1
    error('This function is intended for continuous data!')
end

if strcmp(mode,'remove')
    ksamples = 1:D.nsamples;
    rsamples=[];
    for a = 1:size(segments,1)
        rsamples = [rsamples D.indsample(segments(a,1)):D.indsample(segments(a,end))];
    end
    ksamples(rsamples)=[];
elseif strcmp(mode,'keep')
    rsamples = 1:D.nsamples;
    ksamples = [];
    for a=1:size(segments,1)
        ksamples = [ksamples D.indsample(segments(a,1)):D.indsample(segments(a,end))];
    end
            
    rsamples(ksamples) = [];
end

fname = D.fname;
D=wjn_spm_copy(D.fullfile,['a' fname]);del=D;
try
anames = fieldnames(D.analog);
for a = 1:length(anames)
    D.analog.(anames{a})(rsamples)=nan;
end
catch
    disp('no analog channels found')
end
if length(size(D))==4
    nd = D(:,:,ksamples,:);
    D(:,:,rsamples,:) = nan;
else
    try
        nd = D(:,ksamples,:);
        D(:,rsamples,:) = nan;
    catch
         
    end
end
save(D)
if length(size(nd))==3
    s = [size(nd) 1];
else
    s=size(nd);
end


if length(s)==2;
    s(3) = 1;
end
nD=clone(D,['r' D.fname],s);
nD=timeonset(nD,0);
% keyboard
nD(:,:,:,:) = nd(:,:,:,:);
try
for a = 1:length(anames)
    nD.analog.(anames{a})(rsamples)=nan;
end
catch
end

BIDS_struct_save = struct;
BIDS_struct_save.data = nd;
BIDS_struct_save.labels = nD.chanlabels;
BIDS_struct_save.fsample = nD.fsample;
save(strcat('BIDS_save_', extractBefore(f_name_save, 'Poly5'), 'mat'), 'BIDS_struct_save')
%save(nD)

% save raw file since spm files 1. cannot be read in python 
% 2. .dat files are read as raw_byte streams (not in array shapes) 
% 3. there exists no spm eeg python reader, and the file_array type in 
% Matlab is of SPM type 

D=nD;
del.delete()