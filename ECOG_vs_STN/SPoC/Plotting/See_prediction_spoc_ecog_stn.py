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
import IO
import os
from scipy import signal

from matplotlib.backends.backend_pdf import PdfPages
plt.close('all')
settings = {}
settings['num_patients']=['000', '001', '004', '005', '006', '007', '008', '009', '010', '013', '014']
settings['BIDS_path'] = "/mnt/Datos/BML_CNCRS/Data_BIDS/"
settings['out_path'] = "/mnt/Datos/BML_CNCRS/Spoc/ECoG_STN/"
#%%
def xcorr(x, y, normed=True, detrend=False, maxlags=10):
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed=True
    # Returns inner products when normed=False
    # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)
    # Optional detrending e.g. mlab.detrend_mean

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')
    
    if detrend:
        import matplotlib.mlab as mlab
        x = mlab.detrend_mean(np.asarray(x)) # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))
    
    c = np.correlate(x, y, mode='full')

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    return lags, c
#%%
auc_all_ecog=[]
methods= ["CON", "IPS"]
signals=["ECOG", "STN"]
address='/mnt/Datos/BML_CNCRS/Spoc/ECoG_STN/LM_Out_SPoC/'
pp = PdfPages('output_plot_lm_bop.pdf', keep_empty=False)

r2_stn_ecog=np.zeros((2,2,16))    
TRAINING=False
for m, eeg in enumerate(signals):    
    for n, method in enumerate(methods):
        r2_all=[]

        for s in range(len(settings['num_patients'])):
   
            subject_path=settings['BIDS_path'] + 'sub-' + settings['num_patients'][s]
            subfolder=IO.get_subfolders(subject_path)
            for ss in range(len(subfolder)):
                
                result=np.load(address + settings['num_patients'][s]+'predictions_'+eeg+'_tlag_bopt_'+str(subfolder[ss])+'.npy',allow_pickle=True)
                result=result.tolist()
                        
                score_te=result['score_te'][method]
                score_tr=result['score_tr'][method]
                xc=[]
                
               
                ind_best=np.argmax(score_te)
                ytrue_=result["y_test"]["CON"][ind_best]
                ypre_=result["y_pred_test"]["CON"][ind_best]
                
                fig0, ax0 = plt.subplots(1, 1, figsize=[10, 4])
               
                #times = raw.times[meg_epochs.events[:, 0] - raw.first_samp]
                ax0.plot(ypre_, color='b', label='Predicted mov')
                ax0.plot(ytrue_, color='r', label='True mov')
                ax0.set_xlabel('Time (s)')
                ax0.set_ylabel('Movement')
                ax0.set_title('SPoC mov Predictions')
                ax0.text(0.33, 0.9, 'R2={:0.02f}'.format(score_te[ind_best]),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax0.transAxes,fontsize=12) 
                fig0.suptitle(eeg+'-Subject_'+ settings['num_patients'][s], fontsize=14, fontweight='bold')
                plt.legend()
                plt.savefig(pp, format='pdf')
                #
                                
                lags, c=xcorr(ypre_,ytrue_, normed=True, detrend=False, maxlags=None)
                                   
                maxi=np.argmax(c)

                fig, ax = plt.subplots()

                ax.plot(lags,c)
                # ax.fill_between(lags,xc.mean(axis=0)-xc.std(axis=0),xc.mean(axis=0)+xc.std(axis=0),alpha=.1)   
                ax.vlines(lags[maxi], ymin=0,ymax=1, colors='k', linestyles='dashed', transform=ax.get_xaxis_transform())
                
                ax.set_xlabel('Time lags')
                ax.set_ylabel('Corr coef.')
                ax.set_title('Xcorr with SPOC prediction')
                ax.text(0.33, 0.9, 'xcorr={:0.02f}'.format(c[maxi]),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,fontsize=12) 
                ax.text(0.33, 0.85, 'R^2={:0.02f}'.format(score_te[ind_best]),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,fontsize=12) 
                ax.text(0.33, 0.8, 'lag={:0.02f}'.format(lags[maxi]),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,fontsize=12) 
                fig.suptitle(eeg+'-Subject_'+ settings['num_patients'][s], fontsize=14, fontweight='bold')
                
                plt.savefig(pp, format='pdf')
                r2_all.append(np.mean(score_te))    
        r2_stn_ecog[m,n]=r2_all
        
       
    
#%%

fig, axes = plt.subplots(ncols=2, sharey=True)
fig.subplots_adjust(wspace=0)
cont=0
for ax, name in zip(axes, ['ECOG','STN']):
    
    bp=ax.boxplot(r2_stn_ecog[cont].T)

    ax.set(xticklabels=['CON', 'IPS'], xlabel=name)
    if name=='ECOG': 
        ax.set_ylabel('$R^2$',fontsize=14)

    ax.margins(0.05) # Optional
    for box in bp['boxes']:
        box.set(linewidth=2)
    cont+=1
pp.close()

plt.close("all")
#%%
# fig = plt.figure(1, figsize=(9, 6))
# #Create an axes instance
# ax = fig.add_subplot(131)
# ax.stem(result["coef"]["lm"][0])
# ax.set_title("lm", fontsize=14)
# ax.set_ylabel('coef. value',fontsize=14)
# ax.set_xticks(range(16))

# ax = fig.add_subplot(132)
# ax.stem(result["coef"]["lasso"][0])
# ax.set_title("lasso",fontsize=14)
# ax.set_xticks(range(16))

# ax = fig.add_subplot(133)
# ax.stem(result["coef"]["enet"][0])
# ax.set_title("enet",fontsize=14)
# fig.suptitle('SPoC', fontsize=16, fontweight='bold')
# ax.set_xticks(range(16))
