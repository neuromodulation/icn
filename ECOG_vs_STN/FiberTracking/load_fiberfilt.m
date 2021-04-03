addpath('C:\Users\ICN_admin\Documents\wjn_toolbox')
addpath(genpath('C:\Users\ICN_admin\Documents\leaddbs'))
addpath(genpath('C:\Users\ICN_admin\Documents\spm12'))

% created with fiberfiltering.mat
load('ECOG_XGBOOST.fibfilt','-mat')

resultfig = ea_mnifigure;

ea_discfiberexplorer('ECOG_XGBOOST.fibfilt',resultfig)

