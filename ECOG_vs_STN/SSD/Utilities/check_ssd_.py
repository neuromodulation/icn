#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:33:17 2020

@author: victoria
"""
#check ssd API implementation for epoched and raw data
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/victoria/icn/icn_m1')
import IO
import preprocessing
import matplotlib.pyplot as plt
from ssd import  SSD
import mne
from mne.datasets.fieldtrip_cmc import data_path
from mne.utils import _time_mask
import pickle 
import numpy as np
import locale
locale.setlocale(locale.LC_ALL, "en_US.UTF-8") #needed for local machine in spanish
# plt.close("all")
#%%
def freq_mask(freqs, fmin, fmax):
    """convenience function to select frequencies"""
    return _time_mask(freqs, fmin, fmax)
#%% for raw data
# Define parameters
# vhdr_file = '/mnt/Datos/BML_CNCRS/Data_BIDS_new/sub-000/ses-right/ieeg/sub-000_ses-right_task-force_run-0_ieeg.vhdr'
# bv_raw, ch_names = IO.read_BIDS_file(vhdr_file)
# dat_ECOG, Nan =preprocessing.rereference(run_string=vhdr_file[:-10], bv_raw=bv_raw, get_ch_names=False, get_cortex_subcortex=True)
# sf=IO.read_run_sampling_frequency(vhdr_file)[0]
# used_channels = IO.read_M1_channel_specs(vhdr_file[:-10])
# picks=used_channels["cortex"]
# channels_ecog=[ch_names[i] for i in picks] 
# info_ecog = mne.create_info(ch_names=channels_ecog, sfreq=sf, ch_types='ecog')  
# raw= mne.io.RawArray(dat_ECOG, info_ecog)   

fname = data_path() + '/SubjectCMC.ds'
raw = mne.io.read_raw_ctf(fname)
raw.crop(50., 250.).load_data()  # crop for memory purposes      
picks=mne.pick_types(raw.info, meg=True, eeg=False, ref_meg=False)
sf=raw.info['sfreq']

freqs_sig = 30, 32
freqs_noise = 28, 34
freqs_noise2 = 9, 21

#%%
# ssd = SSD(filt_params_signal=dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
#                                   l_trans_bandwidth=1, h_trans_bandwidth=1,
#                                   fir_design='firwin'),\
#           filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
#                                   l_trans_bandwidth=1, h_trans_bandwidth=1,
#                                   fir_design='firwin'), 
#           filt_params_noise_stop=dict(l_freq=freqs_noise2[1], h_freq=freqs_noise2[0],
#                                   l_trans_bandwidth=1, h_trans_bandwidth=1,
#                                   fir_design='firwin'),
#           sampling_freq=sf, picks=picks, rank="full")
    
ssd = SSD(filt_params_signal=dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1,
                                  fir_design='firwin'),\
          filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1,
                                  fir_design='firwin'), 
          sampling_freq=sf, picks=picks, rank="full", n_fft=4096)    
#%%
ssd.fit(raw.copy().crop(0, 120))
#%%

ssd_sources = ssd.transform(raw)
#%%
psd, freqs = mne.time_frequency.psd_array_welch(
    ssd_sources, sfreq=raw.info['sfreq'], n_fft=4096)
# psd, freqs = mne.time_frequency.psd_array_welch(
#     raw.get_data(), sfreq=raw.info['sfreq'], n_fft=int(np.ceil(raw.info['sfreq']/2)))
#%%
spec_ratio = ssd.spec_ratio

sorter = ssd.sorter_spec

# plot spectral ratio (see Eq. 24 in Nikulin 2011)
plt.figure()
plt.plot(spec_ratio, color='black')
plt.plot(spec_ratio[sorter], color='orange', label='sorted eigenvalues')
plt.xlabel("Eigenvalue Index")
plt.ylabel(r"Spectral Ratio $\frac{P_f}{P_{sf}}$")
plt.legend()
plt.axhline(1, linestyle='--')

# Let's also look at the power spectrum of that source and compare it 
# to the power spectrum of the source with lowest SNR.

below50 = freq_mask(freqs, 0, 200)
bandfilt = freq_mask(freqs, freqs_sig[0],freqs_sig[1])

plt.figure()
plt.loglog(freqs[below50], psd[0, below50], label='max SNR')
plt.loglog(freqs[below50], psd[-1, below50], label='min SNR')
plt.loglog(freqs[below50], psd[:, below50].mean(axis=0), label='mean')
plt.fill_between(freqs[bandfilt], 0, 100,\
                color='green', alpha=0.5)
plt.xlabel("log(frequency)")
plt.ylabel("log(power)")
plt.legend()

#%%
pattern=mne.EvokedArray(data=ssd.patterns_[:4].T,info=mne.pick_info(raw.info, ssd.picks_))
pattern.plot_topomap(units=dict(mag='A.U.'),
                     time_format='')

#%%
X_dnoised=ssd.apply(raw)
#%%
# ssd_denoised=ssd.apply(raw)
# plt.figure()
# plt.psd(mne.filter.filter_data(ssd_denoised[10], raw.info['sfreq'],l_freq=freqs_sig[0], h_freq=freqs_sig[1],
#                                   l_trans_bandwidth=1, h_trans_bandwidth=1,
#                                   fir_design='firwin'))
# plt.psd(mne.filter.filter_data(raw.get_data()[picks[10]], raw.info['sfreq'],l_freq=freqs_sig[0], h_freq=freqs_sig[1],
#                                   l_trans_bandwidth=1, h_trans_bandwidth=1,
#                                   fir_design='firwin'))
# #%% epochs
# file_name='/mnt/Datos/BML_CNCRS/Spoc/ECOG_epochs_wofb_sub_000_sess_right_run_0.p'
# with open(file_name, 'rb') as handle:
#     sub_ = pickle.load(handle)                   
# data=sub_['epochs']
# label_ips=sub_['label_ips']
# label_con=sub_['label_con']
# data=np.squeeze(data)

# # ssd = SSD(filt_params_signal=dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
# #                                   l_trans_bandwidth=1, h_trans_bandwidth=1,
# #                                   fir_design='firwin'),\
# #           filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
# #                                   l_trans_bandwidth=1, h_trans_bandwidth=1,
# #                                   fir_design='firwin'), sampling_freq=1000.0)
# #%%
# ssd.fit(data)

# #%%
# ssd_sources = ssd.transform(data)
# #%%
# spec_ratio = ssd.spec_ratio

# sorter = ssd.sorter_spec

# # plot spectral ratio (see Eq. 24 in Nikulin 2011)
# plt.figure()
# plt.plot(spec_ratio, color='black')
# plt.plot(spec_ratio[sorter], color='orange', label='sorted eigenvalues')
# plt.xlabel("Eigenvalue Index")
# plt.ylabel(r"Spectral Ratio $\frac{P_f}{P_{sf}}$")
# plt.legend()
# plt.axhline(1, linestyle='--')
# #%%
# # Let's investigate spatila filter with max power ratio.
# # We willl first inspect the topographies.
# # According to Nikulin et al 2011 this is done. 
# # by either inverting the filters (W^{-1}) or by multiplying the noise
# # cov with the filters Eq. (22) (C_n W)^t.
# # We rely on the inversion apprpach here.


# max_idx = spec_ratio.argsort()[::-1][:4]


# psd, freqs = mne.time_frequency.psd_array_welch(
#     ssd_sources, sfreq=raw.info['sfreq'], n_fft=int(np.ceil(1000/2)))
# below50 = freq_mask(freqs, 0, 200)
# bandfilt = freq_mask(freqs, freqs_sig[0],freqs_sig[1])
# psd=psd.mean(axis=0)
# plt.figure()
# plt.loglog(freqs[below50], psd[max_idx[0], below50], label='max SNR')
# plt.loglog(freqs[below50], psd[-1, below50], label='min SNR')
# plt.loglog(freqs[below50], psd[:, below50].mean(axis=0), label='mean')
# plt.fill_between(freqs[bandfilt], 0, 100,\
#                 color='green', alpha=0.5)
# plt.xlabel("log(frequency)")
# plt.ylabel("log(power)")
# plt.legend()