%% Converting abagen expression map of 1 gene of interest to nifti
% 08.04.2022 - JVH

%% load the atlas you want to apply on the gene sampling map created by abagen toolbox
nii_atlas = ea_load_nii('C:\Users\Jonathan\Documents\DATA\ATLAS_creation\compound_atlas_HCPex_SUIT_ABGT.nii');
gene_expression_map_fname = 'C:\Users\Jonathan\Documents\PYCHARM\Python\abagen_HCPex_SUIT_ABGT_expression_map.csv';

%% the genetic map is so large, that csvread does not work. Use datastore
ds = datastore(gene_expression_map_fname, 'MissingValue',0); 
% Choose data of interest and data types
ds.SelectedVariableNames = {'label','CSF1R'};
gene_data = readall(ds);

%% here, we want to extract CSF1R and apply its values to the atlas
% we use the atlas, change its name and its voxel intensities to those of
% CSF1R
nii_atlas.fname = 'expression_CSF1R.nii';
nii_atlas.img = changem(nii_atlas.img,gene_data.('CSF1R'),gene_data.('label'));

% change the planar info so the raw values remain. The 3 element depends on
% atlas used
nii_atlas.pinfo = [0;0;352];

%% write out the expression map
ea_write_nii(nii_atlas);

%% check whether the values correspond
nii_expression_CSF1R = ea_load_nii('expression_CSF1R.nii');
unique(nii_expression_CSF1R.img)
%spm_imcalc(nii_atlas.fname,'gene_map_CSF1R.nii','(i1>0).*(i2>0)')
