%% Apply a new atlas to an image
% Here the atlas HCPex_SUIT_AGBT was applied to the PET scan,
% which means that the scan was masked.
% Each parcel of the new image is now the average voxel intensity
% of the corresponding parcel on the PET scan.
% 08.04.2022 - JVH

%% SPM is needed for Lead DBS, which we use to read and write niftis
addpath('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\spm12')
addpath(genpath('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\lead'))

%% load the atlas, the PET scan of interest, and define the output
nii_atlas = ea_load_nii('C:\Users\Jonathan\Documents\DATA\ATLAS_creation\compound_atlas_HCPex_SUIT_ABGT.nii');
nii_PET = ea_load_nii('C:\Users\Jonathan\Documents\CODE\hansen_receptors\data\PET_nifti_images\DAT_fepe2i_hc6_sasaki.nii.gz');
nii_out = nii_atlas;

%% read in the accopanying area label table and define the output
areas = readtable('C:\Users\Jonathan\Documents\DATA\ATLAS_creation\compound_atlas_HCPex_SUIT_ABGT.txt');
areas_out = table({'id'},{'intensity'},{'label'});


%% set the output to zero (or let it take the values of the PET scan, when you do not want to average)
%nii_out.img = nii_PET.img(nii_atlas.img>0); %images MUST have the same dimension
nii_out.img = zeros(nii_out.dim); % set the out image to zeros

%% each parcel of the atlas, apply it to the PET scan, then average and sum it to the output
for p = 1:length(unique(nii_atlas.img))-1
    parcel = zeros(nii_out.dim);
    avg_intensity =  mean(nii_PET.img(nii_atlas.img==p));
    parcel(nii_atlas.img==p) = avg_intensity;
    nii_out.img = nii_out.img + parcel;
    
    % add to the tabulation
    areas_out = [areas_out;{p,avg_intensity,areas.Var2{p}}];
end

%% pint out the output
nii_out.fname = 'compound_atlas_DAT_fepe2i_hc6_sasaki.nii';
nii_out.pinfo = [0;0;352];
ea_write_nii(nii_out);

%% print out the accompanying text file

writetable(areas_out,'compound_atlas_DAT_fepe2i_hc6_sasaki.csv','Delimiter',',')
