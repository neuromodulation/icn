%% Converting abagen expression map of 1 gene of interest to nifti
% 08.04.2022 - JVH

%% SPM is needed for Lead DBS, which we use to read and write niftis
addpath('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\spm12')
addpath(genpath('C:\Users\Jonathan\Documents\MATLAB\add_on_Matlab\lead'))


nii_atlas = ea_load_nii('C:\Users\Jonathan\Documents\DATA\ATLAS_creation\compound_atlas_HCPex_SUIT_ABGT.nii');
nii_PET = ea_load_nii('C:\Users\Jonathan\Documents\CODE\hansen_receptors\data\PET_nifti_images\DAT_fepe2i_hc6_sasaki.nii.gz');
nii_out = nii_atlas;

areas = readtable('C:\Users\Jonathan\Documents\DATA\ATLAS_creation\compound_atlas_HCPex_SUIT_ABGT.txt');
areas_out = table({'id'},{'intensity'},{'label'});


nii_out.img = nii_PET.img(nii_atlas.img>0); %images MUST have the same dimension
nii_out.img = zeros(nii_out.dim); % set the out image to zeros

for p = 1:length(unique(nii_atlas.img))-1
    parcel = zeros(nii_out.dim);
    avg_intensity =  mean(nii_PET.img(nii_atlas.img==p));
    parcel(nii_atlas.img==p) = avg_intensity;
    nii_out.img = nii_out.img + parcel;
    
    % add to the tabulation
    areas_out = [areas_out;{p,avg_intensity,areas.Var2{p}}];
end

nii_out.fname = 'compound_atlas_DAT_fepe2i_hc6_sasaki.nii';
nii_out.pinfo = [0;0;352];
ea_write_nii(nii_out);

%% print out the accompanying text file

writetable(areas_out,'compound_atlas_DAT_fepe2i_hc6_sasaki.csv','Delimiter',',')
