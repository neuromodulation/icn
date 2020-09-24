import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
import scipy
import mne
import os
import pandas as pd
import pickle
import numpy as np
import mne
import scipy
from scipy import io



faces = io.loadmat('faces.mat')
Vertices = io.loadmat('Vertices.mat')
grid = io.loadmat('grid.mat')['grid']
stn_surf = io.loadmat('STN_surf.mat')
x_ = stn_surf['vertices'][::2,0]
y_ = stn_surf['vertices'][::2,1]
x_ecog = Vertices['Vertices'][::1,0]
y_ecog = Vertices['Vertices'][::1,1]
x_stn = stn_surf['vertices'][::1,0]
y_stn = stn_surf['vertices'][::1,1]

def plot_all_in_one(coord_ECOG, coord_STN, p_ECOG_CON, p_ECOG_IPS, p_STN_CON, p_STN_IPS, unit="", meas=""):
    height_STN = 1
    height_ECOG = 2.5*height_STN
    fig, axes = plt.subplots(2,2, facecolor=(0,0,0), gridspec_kw={'height_ratios': [height_ECOG, height_STN]}, \
                             figsize=(14,9))#, dpi=300)
    for idx in range(2):
        axes[0, idx].scatter(x_ecog, y_ecog, c="gray", s=0.001)
        axes[1, idx].scatter(x_stn, y_stn, c="gray", s=0.001)

        if idx == 0: # CON
            pos_ecog = axes[0, idx].scatter(np.array(coord_ECOG)[:,0], \
                                        np.array(coord_ECOG)[:,1], c=p_ECOG_CON, s=10, alpha=0.8, cmap='viridis')

            c = p_STN_CON
            axes[0, idx].set_title(meas+'\ncontralateral performance', color='white')
        if idx == 1:
            pos_ecog = axes[0, idx].scatter(np.array(coord_ECOG)[:,0], \
                                        np.array(coord_ECOG)[:,1], c=p_ECOG_IPS, s=10, alpha=0.8, cmap='viridis')
            axes[0, idx].set_title(meas+'\nipsilateral performance', color='white')
            c = p_STN_IPS
        cbar_ecog = fig.colorbar(pos_ecog, ax=axes[0, idx]); #pos_ecog.set_clim(0,0.7);
        cbar_ecog.set_label(unit, color="white")
        cbar_ecog.ax.tick_params(axis='y', color='white')
        cbar_ecog.ax.set_yticklabels(labels=np.round(cbar_ecog.get_ticks(),2),color='white')
        cbar_ecog.outline.set_edgecolor('white')


        #if len(c) == 4:
        #    c_restructure = [c[0], (c[0]+c[1])/2, (c[1]+c[2])/2, c[2]]
        #elif len(c) == 8:
        #    c_restructure = [c[0], (c[0]+c[1])/2, (c[1]+c[2])/2, c[2],
        #                    c[4], (c[4]+c[5])/2, (c[5]+c[6])/2, c[6]]
        pos_stn = axes[1, idx].scatter(np.array(coord_STN)[:,0], np.abs(np.array(coord_STN)[:,1])*-1, c=c, s=10, alpha=0.8, cmap='viridis')
        cbar_stn = fig.colorbar(pos_stn, ax=axes[1, idx]);
        #pos_stn.set_clim(0,0.7); cbar_stn.remove()
        cbar_stn.outline.set_edgecolor('white')
        cbar_stn.ax.set_yticklabels(labels=np.round(cbar_stn.get_ticks(),2),color='white')
        cbar_stn.set_label(unit, color="white")

        axes[0, idx].axes.set_aspect('equal', anchor='C')
        axes[0, idx].set_facecolor((0,0,0))
        axes[1, idx].axes.set_aspect('equal', anchor='C')
        axes[1, idx].set_facecolor((0,0,0))
