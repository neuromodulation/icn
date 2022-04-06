% This script can remove (and add) parcells to a parcellation atlas
% Here, we remove 'Putamen','Accumbens','Caudate' from the HCPex Atlas

%% SPM is needed for Lead DBS, which we use to read and write niftis
addpath('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\spm12')
addpath(genpath('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\lead'))
%% WJN toolbox is required just for the ci function for convenience
addpath('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\wjn_toolbox')


%% Read in the Parcellation table 
areas = readtable('HCPex (Huang 2021).txt');

%% Find the indices of the regions we want to exclude
indices = areas.Var1(ci({'Putamen','Accumbens','Caudate'},areas.Var2));
% Note that my self-written ci function finds all entities containing the
% search terms, which is why both left and rights hemisphere entries are
% indexed. You could replace this part by a regex command. 


%% Read in the nifti file
nii = ea_load_nii('C:\Users\Jonathan\Documents\DATA\ATLAS_creation\HCPex (Huang 2021).nii');


% %% Loop through the indices of the regions we want to remove and replace them with 0
% for a = 1:length(indices)
%     nii.img(nii.img(:)==indices(a)) = 0;
% end

%% Remove the unwanted areas and collaps the numbers above
% To close the gaps, the highest indices need to be substracted by the number of gaps prior to it

old_vector =  table2array(areas(:,1));
% the shift tracks how the collapsing of the indices look like
shift = zeros(length(old_vector),1);

% each index strictly above the area to be remove needs to be -1
for a = 1:length(indices)
    shift = shift - double(old_vector>indices(a));
end
% on the areas to be removed, the shift equals the substraction of its own index
for a = 1:length(indices)
    shift(old_vector==indices(a),:) = -indices(a);
end

% the new values are the old values plus the shift (note: the shift
% contains neg values)
target_vector = old_vector+shift;

% change the values in the images from old to target vector
nii.img = changem(nii.img,target_vector,old_vector);

%% create the text file
areas.Var1 = target_vector;
areas(areas.Var1==0,:) = [];


%% IMPORTANT: Change the filename to not overwrite the original nifti 
nii.fname = 'HCPex (Huang 2021)_no striatum_renumbered.nii';


%% Write the nii variable to a nifti file
ea_write_nii(nii);
writetable(areas,'HCPex (Huang 2021)_no striatum_renumbered.txt','Delimiter',' ')
% remove the column names

%% Check that it worked as expected
spm_check_registration(nii.fname)