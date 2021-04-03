addpath('C:\Users\ICN_admin\Documents\wjn_toolbox')
addpath(genpath('C:\Users\ICN_admin\Documents\leaddbs'))
addpath(genpath('C:\Users\ICN_admin\Documents\spm12'))

close all, clear all, clc

%% READ THE TABLE WITH MNI COORDINATES AND VARIABLES OF INTEREST FOR ECOG AND STN:
T=readtable('C:\Users\ICN_admin\Documents\icn\ECOG_vs_STN/Results_Analysis/df_all.csv');
fname = 'ECOG';
iecog = ci('ECOG',T.ch);
Tecog=T(iecog,:);

istn = ci('STN',T.ch);
Tstn=T(istn,:);

%% RUN FILTERING FOR ECOG
mkdir ECOG
cd('ECOG')
fname = 'ECOG_XGBOOST';
mni_ecog = [-abs(Tecog.x) Tecog.y Tecog.z];

% Create region of interest nifti files
 ecog_files={};ecog_group = [];roi_radius = 10;
for a=1:size(mni_ecog,1)
    ecog_files{a,1}=wjn_spherical_roi([num2str(a) '_sub-' num2str(Tecog.sub(a)) '_' Tecog.ch{a} '.nii'],mni_ecog(a,:),roi_radius,fullfile(spm('dir'),'canonical','avg152T1.nii'));
    ecog_group(a) = Tecog.sub(a)+1;
end

% Create a Pseudo Lead-Group structure
M.pseudoM = 1; % Declare this is a pseudo-M struct, i.e. not a real lead group file
M.ROI.list=ecog_files; % enter the new files creates from MNI coordinates here
M.ROI.group=ones(length(ecog_files),1);

M.ROI.group=ecog_group;

M.clinical.labels={'r2_con','r2_norm'}; % how will variables be called
M.clinical.vars{1}=Tecog.r2_con; % enter a variable of interest - entries correspond to nifti files
M.clinical.vars{2}=wjn_gaussianize(Tecog.r2_con);
M.guid=fname; % give your analysis a name
save(fname,'M'); % store data of analysis to file

resultfig = ea_mnifigure; % Create empty 3D viewer figure

% Open up the Fiber Filtering Explorer
ea_discfiberexplorer(fullfile(pwd,fname),resultfig);

% see screen capture
%% RUN FILTERING FOR STN

mkdir STN
cd('STN')
fname = 'STN_XGBOOST';
mni_stn = [-abs(Tstn.x) Tstn.y Tstn.z];

% Create region of interest nifti files
 stn_files={};stn_group = [];roi_radius = 5;
for a=1:size(mni_stn,1)
    stn_files{a,1}=wjn_spherical_roi([num2str(a) '_sub-' num2str(Tstn.sub(a)) '_' Tstn.ch{a} '.nii'],mni_stn(a,:),roi_radius,fullfile(spm('dir'),'canonical','avg152T1.nii'));
    stn_group(a,1) = Tstn.sub(a)+1;
end

% Create a Pseudo Lead-Group structure
M.pseudoM = 1; % Declare this is a pseudo-M struct, i.e. not a real lead group file
M.ROI.list=stn_files; % enter the new files creates from MNI coordinates here
M.ROI.group=ones(length(stn_files),1);
M.clinical.labels={'r2_con','r2_norm'}; % how will variables be called
M.clinical.vars{1}=Tstn.r2_con; % enter a variable of interest - entries correspond to nifti files
M.clinical.vars{2}=wjn_gaussianize(Tstn.r2_con);
M.guid=fname; % give your analysis a name
save(fname,'M'); % store data of analysis to file

resultfig = ea_mnifigure; % Create empty 3D viewer figure

% Open up the Fiber Filtering Explorer
ea_discfiberexplorer(fullfile(pwd,fname),resultfig);


%%