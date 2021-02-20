addpath('C:\Users\ICN_admin\Documents\wjn_toolbox')
addpath(genpath('C:\Users\ICN_admin\Documents\leaddbs'))
addpath(genpath('C:\Users\ICN_admin\Documents\spm12'))

%% Decoding performance interpolated to Surface
%root = 'D:\Dropbox (Brain Modulation Lab)\Shared Lab Folders\CRCNS\MOVEMENT DATA';
root = 'C:\Users\ICN_admin\Dropbox (Brain Modulation Lab)\Shared Lab Folders\CRCNS\MOVEMENT DATA\AvgR_Fz';
csvfile = 'C:\Users\ICN_admin\Documents\icn\ECOG_vs_STN\Results_Analysis\df_all.csv';

[files,folders] = wjn_subdir(fullfile(root,'sub*ECOG*Z*.nii'));

T=readtable(csvfile);
istn = ci('STN',T.ch);
iecog = ci('ECOG',T.ch);

stn = [ T.r2_con(istn) -abs(T.x(istn)) T.y(istn) T.z(istn)];
figure
wjn_plot_surface('STN.surf.gii')
hold on
wjn_plot_surface('STN.surf.gii',stn)
hold on
caxis([0 .6])
caire = [-12.58, -13.41, -5.87];
plot3(caire(1),caire(2),caire(3),'linestyle','none','marker','x','MarkerEdgeColor','r','Markersize',40)
alpha .5
myprint('STN_overlay05')
alpha 1
myprint('STN_overlay1')
%

ecog = [ T.r2_con(iecog) -abs(T.x(iecog)) T.y(iecog) T.z(iecog)];
hk = [-37 -25 62];
ctx = load('CortexLowRes_15000V.mat');
ctx.Faces = ctx.Faces_lh;
ctx.Vertices = ctx.Vertices_lh;

figure
wjn_plot_surface(ctx)
hold on
wjn_plot_surface(ctx,ecog)
% plot3(ecog(:,2),ecog(:,3),ecog(:,4),'rx')
hold on
plot3(hk(1),hk(2),hk(3),'linestyle','none','marker','x','MarkerEdgeColor','r','Markersize',40)
caxis([0 .6])
alpha .5
myprint('ECOG_overlay05')
alpha 1
myprint('ECOG_overlay1')

%% Write out spatial interpolation nifti
[files,folders] = wjn_subdir(fullfile(root,'sub*ECOG*Z*.nii'));

T=readtable(csvfile);
istn = ci('STN',T.ch);
iecog = ci('ECOG',T.ch);

stn = [ T.r2_con(istn) -abs(T.x(istn)) T.y(istn) T.z(istn)];

ecog = [ T.r2_con(iecog) -abs(T.x(iecog)) T.y(iecog) T.z(iecog)];
hk = [-37 -25 62];
wjn_heatmap('STN_XGB_performance.nii',stn(:,2:end),stn(:,1),'C:\Users\ICN_admin\Documents\spm12\canonical\single_subj_T1.nii')
wjn_heatmap('ECOG_XGB_performance.nii',ecog(:,2:end),ecog(:,1),'C:\Users\ICN_admin\Documents\spm12\canonical\single_subj_T1.nii')

