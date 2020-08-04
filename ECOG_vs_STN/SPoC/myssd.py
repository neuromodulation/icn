#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:55:45 2020

@author: victoria
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy
# from mne import 
import mne
from scipy import linalg



class SSD(BaseEstimator, TransformerMixin):
    """
    This is a Python Implementation of the ssd function available at 
    https://github.com/svendaehne/matlab_SPoC/tree/master/SSD
    """
  


    def __init__(self, n_components,freq, sampling_freq, denoised=True, return_filtered= True, reg=None, 
                 cov_method_params=None, rank=None):
        # Init default CSP
        if not isinstance(n_components, int):
            raise ValueError('n_components must be an integer.')
        self.n_components = n_components
        
        # make sure FREQS has the correct dimensions
        if (len(freq) !=3 and np.shape(freq)[1] !=2): 
            raise ValueError("freq must be a 3 by 2 matrix, i.e. three bands must be specified!")
      
        #% check the given frequency bands
        signal_band = freq[0] # signal bandpass band
        noise_bp_band = freq[1] #noise bandpass band
        noise_bs_band = freq[2] # noise bandstop band
        #check freq bands
        if (noise_bs_band[0] > signal_band[0] or  noise_bp_band[0] > noise_bs_band[0] or
                signal_band[1] > noise_bs_band[1] or noise_bs_band[1] > noise_bp_band[1]):
            raise ValueError('Wrongly specified frequency bands!\nThe first band (signal band-pass) must be within the third band (noise band-stop) and the third within the second (noise band-pass)!')
    
        self.freq=freq
        self.sampling_freq=sampling_freq
        self.rank = rank
        self.reg = reg    
        self.denoised=denoised
        self.return_filtered=return_filtered
        
        #check flags
        
        if denoised is False and return_filtered is True:
            raise ValueError('return_filtered cannot be True is denoised is Falsed')
            
        
        self.cov_method_params = cov_method_params
        
    def _check_Xy(self, X):
        if X.ndim < 3:
            raise ValueError('X must have at least 3 dimensions.')


    def fit(self, X):
        self._check_Xy(X)

        
        # data X is already filtered in the freq. of interest.
        # compure cov matrix of "signal"
        # the following cope is copied drom mne csp
        n_epochs, n_channels, n_samples = X.shape

        # Estimate single trial covariance
        signal_band = self.freq[0] #signal bandpass band
        #rephase for filtering
        X_aux=np.reshape(X, [n_epochs, n_channels*n_samples])
        X_s=mne.filter.filter_data(X_aux, self.sampling_freq, l_freq=signal_band[0], h_freq=signal_band[1])  
        #reshape to original shape
        X_s=np.reshape(X_s, [n_epochs,n_channels,n_samples])
        covs = np.empty((n_epochs, n_channels, n_channels))
        for ii, epoch in enumerate(X_s):
            covs[ii] = mne.cov._regularized_covariance(
                epoch, reg=self.reg, method_params=self.cov_method_params,
                rank=self.rank)
            
        C_s = covs.mean(0)
    
        #% Covariance matrix for the flanking frequencies (noise)

        noise_bp_band = self.freq[1] #noise bandpass band
        noise_bs_band = self.freq[2] # noise bandstop band
        #rephase for filtering
        X_aux=np.reshape(X, [n_epochs, n_channels*n_samples])
        X_n=mne.filter.filter_data(X_aux, self.sampling_freq, l_freq=noise_bp_band[0], h_freq=noise_bp_band[1])  
        X_n=mne.filter.filter_data(X_n, self.sampling_freq, l_freq=noise_bs_band[1], h_freq=noise_bs_band[0],l_trans_bandwidth=2, h_trans_bandwidth=2)#band stop   
        #reshape to original shape
        X_n=np.reshape(X_n, [n_epochs,n_channels,n_samples])
        # Estimate single trial covariance
        covs_n = np.empty((n_epochs, n_channels, n_channels))
        for ii, epoch in enumerate(X_n):
            covs_n[ii] = mne.cov._regularized_covariance(
                epoch, reg=self.reg, method_params=self.cov_method_params,
                rank=self.rank)

        C_n = covs_n.mean(0)
               
        # solve eigenvalue decomposition
        evals, evecs = linalg.eigh(C_s, C_n)
        evals = evals.real
        evecs = evecs.real
        # sort vectors
        ix = np.argsort(np.abs(evals))[::-1]

        # sort eigenvectors
        evecs = evecs[:, ix].T

        # spatial patterns
        self.patterns_ = linalg.pinv(evecs).T  # n_channels x n_channels
        self.filters_ = evecs  # n_channels x n_channels (each row is a filter)
        
        
                
        return self
 

    def best_component(self,q): 
        W=self.filters_       #each row a filter
        Q25=np.percentile(np.var(W, axis=1),25)
        Q75=np.percentile(np.var(W, axis=1),75)

        thr=Q75+q*(Q75-Q25)
        n_comps=np.where(np.var(W, axis=1)>=thr)[0]
        if len(n_comps)==0:
            n_comps=round(len(W)/2)
        else:
            n_comps=len(n_comps)
        return n_comps

    def transform(self, X):
        """Estimate epochs sources given the SSD filters.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        X : ndarray
            If self.transform_into == 'average_power' then returns the power of
            CSP features averaged over time and shape (n_epochs, n_sources)
            If self.transform_into == 'csp_space' then returns the data in CSP
            space and shape is (n_epochs, n_sources, n_times)
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        if self.n_components==-1:
             n_comps=self.best_component(q=0.01)
             self.n_components=n_comps
           
        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])
        if self.denoised:
        
            pick_patterns = self.patterns_[:self.n_components]
            X=np.asarray([np.dot(pick_patterns.T, epoch) for epoch in X])
            if self.return_filtered:
                #filter data
                n_epochs, n_channels, n_samples = X.shape
                signal_band = self.freq[0] #signal bandpass band
                #rephase for filtering
                X_aux=np.reshape(X, [n_epochs, n_channels*n_samples])
                X_s=mne.filter.filter_data(X_aux, self.sampling_freq, l_freq=signal_band[0], h_freq=signal_band[1])  
                #reshape to original shape
                X=np.reshape(X_s, [n_epochs,n_channels,n_samples])
                
               

        return X
    
 