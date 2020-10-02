#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.linalg import eigh
from mne.io.base import BaseRaw
from mne.epochs import BaseEpochs
from mne.utils import (_validate_type,_time_mask, fill_doc)
from mne.cov import (_regularized_covariance,compute_raw_covariance)
from mne.filter import filter_data
from mne.time_frequency import psd_array_welch

@fill_doc
class SSD(BaseEstimator, TransformerMixin):
    """
    
    This is a Python Implementation of Spatio Spectral Decomposition (SSD) 
    method [1],[2] for both raw and epoched data. This source code is  based on
    the matlab implementation available at 
    https://github.com/svendaehne/matlab_SPoC/tree/master/SSD and the PR MNE-based
    implementation of Denis A. Engemann <denis.engemann@gmail.com>
    
    SSD seeks at maximizing the power at a frequency band of interest while
    simultaneously minimizing it at the flanking (surrounding) frequency bins
    (considered noise). It extremizes the covariance matrices associated to 
    signal and noise.
    
    Cosidering f as the freq. of interest, noise signals can be calculated
    either by filtering the signals in the frequency range [f−Δfb:f+Δfb] and then 
    performing band-stop filtering around frequency f [f−Δfs:f+Δfs], where Δfb
    and Δfs generally are set equal to 2 and 1 Hz, or by filtering 
    in the frequency range [f−Δf:f+Δf] with the following subtraction of 
    the filtered signal around f [1]. 
    
    SSD can either be used as a dimentionality reduction method or a denoised 
    ‘denoised’ low rank factorization method [2].
        

    Parameters
    ----------
    filt_params_signal : dict
        Filtering for the frequencies of interst.
    filt_params_noise  : dict
        Filtering for the frequencies of non-interest.
    sampling_freq : float
        sampling frequency (in Hz) of the recordings.
    estimator : float | str | None (default 'oas')
        Which covariance estimator to use
        If not None (same as 'empirical'), allow regularization for
        covariance estimation. If float, shrinkage is used 
        (0 <= shrinkage <= 1). For str options, estimator will be passed to method 
        to mne.compute_covariance().    
    n_components : int| None (default None)
        The number of components to decompose the signals. 
        If n_components is None no dimentionality reduction is made, and the 
        transformed data is projected in the whole source space.
   picks: array| int | None  (default None)
       Indeces of good-channels. Can be the output of mne.pick_types
 
   sort_by_spectral_ratio: bool (default True)
       if set to True, the components are sorted according to the spectral ratio
       See [1] Nikulin 2011, Eq. (24)
    
   cov_method_params : TYPE, optional
        As in mne.decoding.SPoC 
        The default is None.
   rank : None | dict | ‘info’ | ‘full’
        As in mne.decoding.SPoC 
        This controls the rank computation that can be read from the
        measurement info or estimated from the data. 
        See Notes of mne.compute_rank() for details.The default is None.
        The default is None.

    
    REFERENCES:
    [1] Nikulin, V. V., Nolte, G., & Curio, G. (2011). A novel method for 
    reliable and fast extraction of neuronal EEG/MEG oscillations on the basis 
    of spatio-spectral decomposition. NeuroImage, 55(4), 1528-1535.
    [2] Haufe, S., Dähne, S., & Nikulin, V. V. (2014). Dimensionality reduction
    for the analysis of brain oscillations. NeuroImage, 101, 583-597.
    """
  


    def __init__(self, filt_params_signal, filt_params_noise, sampling_freq, filt_params_noise_stop=None,
                 estimator='oas', n_components=None, picks=None,
                 sort_by_spectral_ratio=True, cov_method_params=None, rank=None):
        
        """Initialize instance"""

        dicts = {"signal": filt_params_signal, "noise": filt_params_noise}
        for param, dd in [('l', 0), ('h', 0), ('l', 1), ('h', 1)]:
            key = ('signal', 'noise')[dd]
            if  param + '_freq' not in dicts[key]:
                raise ValueError(
                    "'%%' must be defined in filter parameters for %s" % key)
            val = dicts[key][param + '_freq']
            if not isinstance(val, (int, float)):
                raise ValueError(
                    "Frequencies must be numbers, got %s" % type(val))
       
          #check freq bands
        if (filt_params_noise['l_freq'] > filt_params_signal['l_freq'] or  
                filt_params_signal['h_freq']>filt_params_noise['h_freq']):
            raise ValueError('Wrongly specified frequency bands!\nThe signal band-pass must be within the t noise band-pass!')

        self.freqs_signal = (filt_params_signal['l_freq'],
                             filt_params_signal['h_freq'])
        self.freqs_noise = (filt_params_noise['l_freq'],
                            filt_params_noise['h_freq'])
        
        
        #the band-pass + stop-band filtering for estimating noise wii be used
        self.filt_params_noise_stop=filt_params_noise_stop
      
        self.filt_params_signal = filt_params_signal
        self.filt_params_noise = filt_params_noise
        self.sort_by_spectral_ratio = sort_by_spectral_ratio
        self.picks_ = picks
        
       
        self.estimator = estimator
        self.n_components = n_components
        self.rank = rank
        self.sampling_freq=sampling_freq      
        self.cov_method_params = cov_method_params
        
    
    def fit(self, inst):
        """Fit"""
        
        
        if isinstance(inst, BaseRaw):
            if self.picks_ is None:
                raise ValueError('picks should be provided')
            self.max_components=len(self.picks_)
            inst_signal = inst.copy()
            inst_signal.filter(picks=self.picks_, **self.filt_params_signal)
            
            #noise
            inst_noise = inst.copy()
            inst_noise.filter(picks=self.picks_, **self.filt_params_noise)
            if self.filt_params_noise_stop is not None:          
                #stop-band filtering after the band-pass filtering
                inst_noise.filter(picks=self.picks_, **self.filt_params_noise_stop)
            else:
                # subtract signal:
                inst_noise._data[self.picks_] -= inst_signal._data[self.picks_]
           
            cov_signal = compute_raw_covariance(
                inst_signal, picks=self.picks_, method=self.estimator, rank=self.rank)
            cov_noise = compute_raw_covariance(inst_noise, picks=self.picks_,
                method=self.estimator, rank=self.rank)
            del inst_noise
            del inst_signal
        else:
            if isinstance(inst, BaseEpochs)  or isinstance(inst, np.ndarray):
            # data X is epoched 
            # part of the following code is copied from mne csp
                n_epochs, n_channels, n_samples = inst.shape
                self.max_components=n_channels
              
                #reshape for filtering
                X_aux=np.reshape(inst, [n_epochs, n_channels*n_samples])
                X_s=filter_data(X_aux, self.sampling_freq, **self.filt_params_signal)  
                
                #rephase for filtering
                X_aux=np.reshape(inst, [n_epochs, n_channels*n_samples])
                X_n=filter_data(X_aux, self.sampling_freq, **self.filt_params_noise)
                if self.filt_params_noise_stop is not None:          
                    #stop-band filtering after the band-pass filtering
                    X_n=filter_data(X_n, self.sampling_freq, **self.filt_params_noise_stop)
                else:
                    # subtract signal:
                    X_n -= X_s
                
                # Estimate single trial covariance
                #reshape to original shape
                X_s=np.reshape(X_s, [n_epochs,n_channels,n_samples])
                covs = np.empty((n_epochs, n_channels, n_channels))
                for ii, epoch in enumerate(X_s):
                    covs[ii] = _regularized_covariance(
                        epoch, reg=self.estimator, method_params=self.cov_method_params,
                        rank=self.rank)
                    
                cov_signal = covs.mean(0)
            
                #% Covariance matrix for the flanking frequencies (noise)
                       
                
                
                
                #reshape to original shape
                X_n=np.reshape(X_n, [n_epochs,n_channels,n_samples])
                # Estimate single trial covariance
                covs_n = np.empty((n_epochs, n_channels, n_channels))
                for ii, epoch in enumerate(X_n):
                    covs_n[ii] = _regularized_covariance(
                        epoch, reg=self.estimator, method_params=self.cov_method_params,
                        rank=self.rank)
        
                cov_noise = covs_n.mean(0)
                       
            
            else:
                raise NotImplementedError()
        
        eigvals_,eigvects_ = eigh(cov_signal.data, cov_noise.data)
        #sort in descencing order
        ix = np.argsort(eigvals_)[::-1]
        self.eigvals_=eigvals_[ix]
        self.filters_ = eigvects_[:, ix]
        #filter are columns
        
        self.patterns_ = np.linalg.pinv(self.filters_)      
             
                
        return self
 
    def spectral_ratio_ssd(self,ssd_sources):
        """Spectral ratio measure for best n_components selection
        See Nikulin 2011, Eq. (24)
        
        Parameters
        ----------
        ssd_sources : data projected on the SSD space. 
        output of transform
       
        """
        psd, freqs = psd_array_welch(
        ssd_sources, sfreq=self.sampling_freq, n_fft=int(np.ceil(self.sampling_freq/2)))
        sig_idx = _time_mask(freqs, *self.freqs_signal)
        noise_idx = _time_mask(freqs, *self.freqs_noise)
        if psd.ndim ==3:
            spec_ratio = psd[:, :,sig_idx].mean(axis=2).mean(axis=0) / psd[:,:, noise_idx].mean(axis=2).mean(axis=0) 

        else:
                    
            spec_ratio = psd[:, sig_idx].mean(axis=1) / psd[:, noise_idx].mean(axis=1)
        sorter_spec = spec_ratio.argsort()[::-1]
        return spec_ratio, sorter_spec

    def transform(self, inst):
        """Estimate epochs sources given the SSD filters.

        Parameters
        ----------
        inst : instance of Raw or Epochs (n_epochs, n_channels, n_times)
            The data to be processed. The instance is modified inplace.
       
        Returns
        -------
        out : instance of Raw or Epochs
            The processed data.
        
        """
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first call fit')
        if isinstance(inst, BaseRaw):
            data=inst.get_data()
            X_ssd=np.dot(self.filters_.T, data[self.picks_])
           
        else:
            if isinstance(inst, BaseEpochs)  or isinstance(inst, np.ndarray):
                
                data=inst    
                #project data on source space
                X_ssd = np.asarray([np.dot(self.filters_.T, epoch) for epoch in data])
        
        if self.sort_by_spectral_ratio:
            self.spec_ratio, self.sorter_spec=self.spectral_ratio_ssd(ssd_sources=X_ssd)
            if isinstance(inst, BaseRaw):
                X_ssd=X_ssd[self.sorter_spec]
            else:
                X_ssd=X_ssd[:,self.sorter_spec,:]
            if self.n_components is None:
                n_components = self.max_components
                return X_ssd
            else:
                n_components = self.n_components
                if isinstance(inst, BaseRaw):
                    X_ssd=X_ssd[:n_components]
                else:
                    X_ssd=X_ssd[:,:n_components,:]
             
                
               

        
    
    def apply(self, inst):
        """
        Remove selected components from the signal.
        This procedure will reconstruct M/EEG signals from which the dynamics 
        described by the excluded components is subtracted (denoised by low-rank factorization). 
        See [2]  Haufe et al. for more information.
        
        The data is processed in place.

        Parameters
        ----------
        inst : instance of Raw or Epochs 
            The data to be processed. The instance is modified inplace.
        n_components : int
            The indices referring to columns in the ummixing matrix. The
            components to be kept.
        
       
        Returns
        -------
        out : instance of Raw or Epochs
            The processed data.
        
        """
        X_ssd=self.transform(inst)
        
        pick_patterns = self.patterns_[:self.n_components].T
        if isinstance(inst, BaseRaw):
            X=np.dot(pick_patterns, X_ssd)

        else:
            if isinstance(inst, BaseEpochs):
                if not isinstance(inst, np.ndarray):
                    raise ValueError("X should be of type ndarray (got %s)." % type(inst))
                X=np.asarray([np.dot(pick_patterns, epoch) for epoch in X_ssd])
        
        return X


             

    def inverse_transform(self):
        """
        Not implemented, see ssd.apply() instead.

        """
        	
        raise NotImplementedError()
    
 