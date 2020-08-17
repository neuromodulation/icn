#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:46:27 2020

@author: victoria
"""
#%%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, roc_auc_score
import sys
sys.path.insert(1, '/home/victoria/icn/icn_m1')
# sys.path.insert(1, 'C:\\Users\Pilin\Documents\GitHub\icn\icn_m1')

import IO
import json
from scipy import stats, signal, io
import os
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import mne
from mne.viz import plot_alignment, snapshot_brain_montage
import time
#%%
plt.close('all')
settings = {}
settings['num_patients']=['000', '001', '004', '005', '006', '007', '008', '009', '010', '013', '014']
settings['BIDS_path'] = "/mnt/Datos/BML_CNCRS/Data_BIDS_new/"
settings['out_path'] = "/mnt/Datos/BML_CNCRS/Spoc/ECoG_STN/LM_Out_SPoC/"
settings['frequencyranges']=[[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]]
#%%
def get_pos(archive, eeg, return_z=False):
    
    coord_path = os.path.join(archive)
    df = pd.read_csv(coord_path, sep="\t")
    
    ch_names=df[df['name'].str.contains(eeg)]["name"]
    ch_names=ch_names.tolist()
    pos_x=df[df['name'].str.contains(eeg)]["x"]
    pos_y=df[df['name'].str.contains(eeg)]["y"]
    pos_z=df[df['name'].str.contains(eeg)]["z"]
    
    if return_z is True:
        return pos_x, pos_y, pos_z
    return pos_x, pos_y

def plot_patterns(archive, subject_id_, session,c_ecog, c_stn):
    height_STN = 1
    height_ECOG = 2.5*height_STN
    rows=2; columns = 1
    fig, axes = plt.subplots(rows,columns, facecolor=(1,1,1), gridspec_kw={'height_ratios': [height_ECOG, height_STN]}, \
                             )#, dpi=700)
        
    axes[0].scatter(x_ecog, y_ecog, c="gray", s=0.0001)
    axes[0].set_axis_off()

    axes[1].scatter(x_stn, y_stn, c="gray", s=0.0001)
    axes[1].set_axis_off()


# axes[y_cnt_ECOG, x_cnt_ECOG].scatter(x_ecog, y_ecog, c="gray", s=0.0001)
# axes[y_cnt_ECOG, x_cnt_ECOG].set_title('sub'+subject_id_, color='white')
    x,y = get_pos(archive, 'ECOG')
    if subject_id_ == '006' and session=='ses-right':
        x,y=x[:28],y[:28]

    
    pos_ecog = axes[0].scatter(x, y, c=c_ecog, s=10)

    # cbar_ecog = fig.colorbar(pos_ecog, ax=axes[0]); pos_ecog.set_clim(0,0.5); 
    # # cbar_ecog.remove()
    
    pos_stn = axes[1].scatter(x_stn, y_stn, c="gray", s=0.0001)
  
    x,y = get_pos(archive, 'STN')
     
    x,y=x[:3], y[:3]

    pos_stn = axes[1].scatter(x, y, c=c_stn, s=10)
 
    # cbar_stn = fig.colorbar(pos_stn, ax=axes[1]); pos_stn.set_clim(0,0.5); 
    # cbar_stn.remove()

    axes[1].axes.set_aspect('equal', anchor='C')
    # axes[1].set_facecolor((1,1,1))
    axes[0].axes.set_aspect('equal', anchor='C')
    axes[0].axes.set_title('Sub'+subject_id_+ '_'+session , size='x-large')

    # axes[0].set_facecolor((1,1,1))

    # axes[4, 3].set_facecolor((0,0,0)); axes[4, 3].set_axis_off()
    # axes[5, 3].set_facecolor((0,0,0)); axes[5, 3].set_axis_off()
    #     #axes[6, 1].set_facecolor((0,0,0)); axes[6, 1].set_axis_off()
    #axes[6, 2].set_facecolor((0,0,0)); axes[6, 2].set_axis_off()
    #axes[6, 3].set_facecolor((0,0,0)); axes[6, 3].set_axis_off()
    #axes[7, 1].set_facecolor((0,0,0)); axes[7, 1].set_axis_off()
    #axes[7, 2].set_facecolor((0,0,0)); axes[7, 2].set_axis_off()
    #axes[7, 3].set_facecolor((0,0,0)); axes[7, 3].set_axis_off()
    

#%%
# setup plot where STN and ECOG is visible 
faces = io.loadmat('/home/victoria/icn/icn_plots/faces.mat')
Vertices = io.loadmat('/home/victoria/icn/icn_plots/Vertices.mat')
grid = io.loadmat('/home/victoria/icn/icn_plots/grid.mat')['grid']
stn_surf = io.loadmat('/home/victoria/icn/icn_plots/STN_surf.mat')
x_ = stn_surf['vertices'][::2,0]
y_ = stn_surf['vertices'][::2,1]
x_ecog = Vertices['Vertices'][::1,0]
y_ecog = Vertices['Vertices'][::1,1]
x_stn = stn_surf['vertices'][::1,0]
y_stn = stn_surf['vertices'][::1,1]
#%%

auc_all_ecog=[]
methods= ["CON", "IPS"]
signals=["ECOG", "STN"]
cont=0
r2_stn_ecog=np.zeros((2,2,16))    
TRAINING=False
    
for s in range(1):
    # pp = PdfPages('ECOG_STN_patterns_PLOT'+settings['num_patients'][s]+'.pdf', keep_empty=False)

    for n, method in enumerate(methods):
        r2_all=[]

        subject_path=settings['BIDS_path'] + 'sub-' + settings['num_patients'][s]
        subfolder=IO.get_subfolders(subject_path)
        #%%
        for ss in range(len(subfolder)):
            fig1, axs1 = plt.subplots(2,1)

            for m, eeg in enumerate(signals): 
                file_name=settings['out_path']+ settings['num_patients'][s]+'predictions_' + eeg +'_tlag_bopt_'+str(subfolder[ss])+'.npy'
                result=np.load(file_name, allow_pickle=True)
                
                
                result=result.tolist()
                
                score_te=result['score_te'][method]
                score_tr=result['score_tr'][method]
                
                ind_best=np.argmax(score_te)
                
                coef=result["coef"][method][ind_best]
                max_coef=np.argmax(coef)
                
                ind_f=int(max_coef/5)
                

                
                axs1[m].stem(coef)
                if m==0: axs1[m].set_xticks([])

                fig1.suptitle(method+'-S'+ settings['num_patients'][s]+'_'+str(subfolder[ss])+'_', fontsize=14, fontweight='bold')
                axs1[m].set_title(eeg)
                axs1[m].set_ylabel("Coef. value")

                if m==1: 
                    axs1[m].set_xlabel('Coef. index')
                    if s==0: 
                        name='/home/victoria/Dropbox/Presentaciones/crcns/figs/coef_'+method+'_s_'+ settings['num_patients'][s]+'.png'
                        plt.savefig(name, bbox_inches='tight', format='png')
                                            
                patterns=result["patterns"][method][ind_best]
                pattern=np.squeeze(patterns[ind_f])
                pattern=pattern[0]
                # pattern -= pattern.mean()
                # ix = np.argmax(abs(pattern))
                # # the parttern is sign invariant.
                # # invert it for display purpose
                # if pattern[ix]>0:
                #     sign = 1.0
                # else:
                #     sign = -1.0
                

                filters=result["filters"][method][ind_best]
                filter1=np.squeeze(filters[ind_f])  
                
               
                archive=settings['BIDS_path']+ 'sub-'+ settings['num_patients'][s]+ '/'+str(subfolder[ss])+ '/ieeg'+ '/sub-'+ settings['num_patients'][s]+ '_electrodes.tsv'
                if eeg== "ECOG":
                    c_ecog=pattern
                else:
                    c_stn=pattern
            
            
            plot_patterns(archive, settings['num_patients'][s], str(subfolder[ss]), c_ecog, c_stn)
            name='/home/victoria/Dropbox/Presentaciones/crcns/figs/patterns_'+method+'_s_'+ settings['num_patients'][s]+'.png'
            plt.savefig(name, bbox_inches='tight', format='png')

            
# pp.close()
                    
