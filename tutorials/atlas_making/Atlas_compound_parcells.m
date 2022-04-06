%% make a compound atlas
% JVH 05-04-2022

%% SPM is needed for Lead DBS, which we use to read and write niftis
addpath('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\spm12')
addpath(genpath('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\lead'))

%% Read in the Parcellation table 
areas1 = readtable('HCPex (Huang 2021)_no striatum_renumbered.txt');
areas2 = readtable('Cerebellum_SUIT (Diedrichsen 2006).txt');
areas3 = readtable('ABGT (He 2020) - 2mm MNI152_only striatum_renumbered.txt');

%% Read in the nifti file
nii1 = ea_load_nii('C:\Users\Jonathan\Documents\DATA\ATLAS_creation\HCPex (Huang 2021)_no striatum_renumbered.nii');
nii2 = ea_load_nii('C:\Users\Jonathan\Documents\DATA\ATLAS_creation\Cerebellum_SUIT (Diedrichsen 2006).nii');
nii3 = ea_load_nii('C:\Users\Jonathan\Documents\DATA\ATLAS_creation\ABGT (He 2020) - 2mm MNI152_only striatum_renumbered.nii');

%% to deal with the issue of partially overlapping parcels, we need to substract masks
% the intersection of nii1 and nii2 is substracted from nii1
% the possible intersection of nii1 and nii3 is substracted from nii1
% it is no possible to have overlap between nii2 and nii3

% flags set to NaNs should be zeroed
flags =struct();
flags.mask = -1;

spm_imcalc(char(nii1.fname,nii2.fname),'intersection_HCPex_no striatum_SUIT.nii','(i1>0).*(i2>0)',flags)
spm_imcalc(char(nii1.fname,nii3.fname),'intersection_HCPex_no striatum_ABGT_only striatum.nii','(i1>0).*(i2>0)',flags)

nii4 = ea_load_nii('C:\Users\Jonathan\Documents\DATA\ATLAS_creation\intersection_HCPex_no striatum_SUIT.nii');
nii5 = ea_load_nii('C:\Users\Jonathan\Documents\DATA\ATLAS_creation\intersection_HCPex_no striatum_ABGT_only striatum.nii');


%% calculate the atlas lengths that need to be added to match the indexing

len_i1=height(areas1);
len_i2=height(areas2);
len_i3=height(areas3);

areas4 = [areas1;areas2;areas3];
areas4.Var1 = [areas1.Var1;areas2.Var1+len_i1;areas3.Var1+len_i1 + len_i2];
% check whether the expression above is generating a continuous vector, because we
% ll do the same for the indices atlas

%% now composite the atlas
% formula:i1-i1.*(i4>0)-i1.*(i5>0) + (i2+(len_i1*(i2>0))) + (i3+((len_i2+len_i3)*(i3>0))) 
% i1-i1.*(i4>0)-i1.*(i5>0) implies substracting the overlap images
% (i2+(len_i1*(i2>0))) implies increase the indices of i2 and masking itself
% (i3+((len_i1+len_i2)*(i3>0))) implies increase the indices of i3 and masking itself

spm_imcalc(char(nii1.fname,nii2.fname,nii3.fname,nii4.fname,nii5.fname),...
    'compound_atlas_HCPex_SUIT_ABGT.nii',...
    'i1 -i1.*(i4>0)-i1.*(i5>0) + (i2+(len_i1*(i2>0))) + (i3+((len_i1+len_i2)*(i3>0)))',...
    flags,...
    len_i1,len_i2,len_i3)


%% print out the accompanying text file

writetable(areas4,'compound_atlas_HCPex_SUIT_ABGT.txt','Delimiter',' ')
% remove the column names if necessary






