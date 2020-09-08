#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:49:39 2020

@author: victoria
"""
"""
===========================================================
Compute Spectro-Spatial Decomposition (SDD) spatial filters
===========================================================
In this example, we will compute spatial filters for retaining
oscillatory brain activity and down-weighting 1/f background signals
as proposed by [1]_.
The idea is to learn spatial filters that separate oscillatory dynamics
from surrounding non-oscillatory noise based on the covariance in the
frequency band of interest and the noise covariance based on surrounding
frequencies.
References
----------
.. [1] Nikulin, V. V., Nolte, G., & Curio, G. (2011). A novel method for
       reliable and fast extraction of neuronal EEG/MEG oscillations on the
       basis of spatio-spectral decomposition. NeuroImage, 55(4), 1528-1535.
"""
# Author: Denis A. Engemann <denis.engemann@gmail.com> 
#         & Victoria Peterson <vpeterson2@mgh.harvard.edu>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
from scipy.linalg import eigh
import numpy as np
import mne
from mne import Epochs
from mne.datasets.fieldtrip_cmc import data_path
from mne.utils import _time_mask
from mne.channels import read_layout
from mne.decoding import TransformerMixin, BaseEstimator
from ssd import  SSD

import locale
locale.setlocale(locale.LC_ALL, "en_US.UTF-8") #needed for local machine in spanish
plt.close("all")
#%%

def freq_mask(freqs, fmin, fmax):
    """convenience function to select frequencies"""
    return _time_mask(freqs, fmin, fmax)

# Define parameters
fname = data_path() + '/SubjectCMC.ds'
raw = mne.io.read_raw_ctf(fname)
raw.crop(50., 250.).load_data()  # crop for memory purposes

freqs_sig = 9, 12
freqs_noise = 8, 13


picks=mne.pick_types(raw.info, meg=True, eeg=False, ref_meg=False)
#%%
ssd = SSD(filt_params_signal=dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1,
                                  fir_design='firwin'),
          filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1,
                                  fir_design='firwin'), 
          sampling_freq=raw.info['sfreq'], 
          picks=picks)
#%%
ssd.fit(raw.copy().crop(0, 120))
#%%
ssd_sources = ssd.transform(raw)
#%%
psd, freqs = mne.time_frequency.psd_array_welch(
    ssd_sources, sfreq=raw.info['sfreq'], n_fft=int(np.ceil(raw.info['sfreq']/2)))
#%%
spec_ratio = ssd.spec_ratio

sorter = ssd.sorter_spec

#%%
# plot spectral ratio (see Eq. 24 in Nikulin 2011)
plt.figure()
plt.plot(spec_ratio, color='black')
plt.plot(spec_ratio[sorter], color='orange', label='sorted eigenvalues')
plt.xlabel("Eigenvalue Index")
plt.ylabel(r"Spectral Ratio $\frac{P_f}{P_{sf}}$")
plt.legend()
plt.axhline(1, linestyle='--')


max_idx = spec_ratio.argsort()[::-1][:4]


# Let's also look at the power spectrum of that source and compare it 
# to the power spectrum of the source with lowest SNR.

below50 = freq_mask(freqs, 0, 50)
bandfilt = freq_mask(freqs, freqs_sig[0],freqs_sig[1])

plt.figure()
plt.loglog(freqs[below50], psd[max_idx[0], below50], label='max SNR')
plt.loglog(freqs[below50], psd[-1, below50], label='min SNR')
plt.loglog(freqs[below50], psd[:, below50].mean(axis=0), label='mean')
plt.fill_between(freqs[bandfilt], 0, 100,\
                color='green', alpha=0.5)
plt.xlabel("log(frequency)")
plt.ylabel("log(power)")
plt.legend()
# We can clearly see that the selected component enjoyes an SNR that is
# way above the average powe spectrum.

# Let's investigate spatila filter with max power ratio.
# We willl first inspect the topographies.
# According to Nikulin et al 2011 this is done. 
# by either inverting the filters (W^{-1}) or by multiplying the noise
# cov with the filters Eq. (22) (C_n W)^t.
# We rely on the inversion apprpach here.

pattern=mne.EvokedArray(data=ssd.patterns_[max_idx, :].T,info=mne.pick_info(raw.info, ssd.picks_))
pattern.plot_topomap(units=dict(mag='A.U.'),
                     time_format='')


# The topographies suggest that we picked up a parietal alpha generator.


#%% check for epoched data
# Filter MEG data to focus on beta band
raw.pick_types(meg=True, ref_meg=False, eeg=False, eog=False)
# Build epochs as sliding windows over the continuous raw file
events = mne.make_fixed_length_events(raw, id=1, duration=.250)

# Epoch length is 1.5 second
meg_epochs = Epochs(raw, events, tmin=0., tmax=1, baseline=None,
                    detrend=1, decim=1)
#%%
X=meg_epochs.get_data()
#%%
ssd.fit(X)
#%%
ssd_sources_epochs = ssd.transform(X)
#%%
psd, freqs = mne.time_frequency.psd_array_welch(
    ssd_sources_epochs, sfreq=raw.info['sfreq'], n_fft=int(np.ceil(raw.info['sfreq']/2)))
#%%
spec_ratio = ssd.spec_ratio

sorter = ssd.sorter_spec

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

#%%
max_idx = spec_ratio.argsort()[::-1][:4]

psd, freqs = mne.time_frequency.psd_array_welch(
    ssd_sources, sfreq=raw.info['sfreq'], n_fft=int(np.ceil(1000/2)))
below50 = freq_mask(freqs, 0, 200)
bandfilt = freq_mask(freqs, freqs_sig[0],freqs_sig[1])
psd=psd.mean(axis=0)
plt.figure()
plt.loglog(freqs[below50], psd[max_idx[0], below50], label='max SNR')
plt.loglog(freqs[below50], psd[-1, below50], label='min SNR')
plt.loglog(freqs[below50], psd[:, below50].mean(axis=0), label='mean')
plt.fill_between(freqs[bandfilt], 0, 100,\
                color='green', alpha=0.5)
plt.xlabel("log(frequency)")
plt.ylabel("log(power)")
plt.legend()
