#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  29  2020

@author: Victoria Peterson
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import mne
mne.set_log_level(verbose='warning') #to avoid info at terminal
from scipy import linalg
from ..io.base import BaseRaw
from ..epochs import BaseEpochs



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
    n_components : int| float
        The number of components to decompose the signals. 
        If n_components is a float number, then the number of component is
        selected based on the threshold criteria proposed in (Eq. 10) [2], 
        being n_components the so defined "q" nonnegative constant.
    freq : list with shape (3,2)
        First index: cut-off frequencies of the freq. band of interest 
        (signal)
        Second index: cut-off frequencies for the lowest and highest 
        frequencies defining flanking intervals.
        Third index: cut-off frequencies for the band-stop filtering of 
        the central frequency process.
    sampling_freq : float
        sampling frequency (in Hz) of the recordings.
    denoised : bool (default True)
        If set to True, the output will be a matrix with the same 
        dimentionality of the original input data but denoised based on the 
        low-rank factorization. For more information about this, please see
        the section of "low-rank factorization" in [2].          
        The default is True.
    return_filtered : bool (default True)
        If denoised is True, the recontructed signal is denoised but not 
        filtered in the freq. band of interest. For further analysis, most 
        probably the denoised recontructed signal would be desired to be used 
        in the especific freq. band of interest. If "return_filtered" is True, 
        then the denoised signal is band-pass filtered in the freq. band given
        in freq[0].
        
        The default is True.
    reg : float | str | None (default None)
        As in mne.decoding.SPoC 
        If not None (same as 'empirical', default), allow regularization for
        covariance estimation. If float, shrinkage is used 
        (0 <= shrinkage <= 1). For str options, reg will be passed to method 
        to mne.compute_covariance().
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
  


    def __init__(self, filt_params_signal, filt_params_noise, sampling_freq,
                 estimator='oas', n_components=None,
                 reject=None,
                 flat=None,
                 picks=None,
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
        self.filt_params_signal = filt_params_signal
        self.filt_params_noise = filt_params_noise
        self.sort_by_spectral_ratio = sort_by_spectral_ratio
        self.picks = picks
        self.estimator = estimator
        self.n_components = n_components
        self.rank = rank
        self.sampling_freq=sampling_freq      
        self.cov_method_params = cov_method_params
        
    # def _check_Xy(self, X):
    #     if X.ndim < 3:
    #         raise ValueError('X must have at least 3 dimensions.')


    def fit(self, inst):
        _validate_type(inst, (BaseRaw, BaseEpochs), 'inst', 'Raw or Epochs')

        
        # data X is epoched 
        # part of the following code is copied from mne csp
        n_epochs, n_channels, n_samples = X.shape

        # Estimate single trial covariance
        signal_band = self.freq[0] #signal bandpass band
        #reshape for filtering
        X_aux=np.reshape(X, [n_epochs, n_channels*n_samples])
        X_s=mne.filter.filter_data(X_aux, self.sampling_freq, l_freq=signal_band[0], h_freq=signal_band[1],method='iir', phase='zero-double')  
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
        X_n=mne.filter.filter_data(X_aux, self.sampling_freq, l_freq=noise_bp_band[0], h_freq=noise_bp_band[1],method='iir', phase='zero-double')  
        X_n=mne.filter.filter_data(X_n, self.sampling_freq, l_freq=noise_bs_band[1], h_freq=noise_bs_band[0],method='iir', phase='zero-double')#band stop   
        #reshape to original shape
        X_n=np.reshape(X_n, [n_epochs,n_channels,n_samples])
        # Estimate single trial covariance
        covs_n = np.empty((n_epochs, n_channels, n_channels))
        for ii, epoch in enumerate(X_n):
            covs_n[ii] = mne.cov._regularized_covariance(
                epoch, reg=self.reg, method_params=self.cov_method_params,
                rank=self.rank)

        C_n = covs_n.mean(0)
        
        
        D, V = linalg.eig(C_s)
        D, V = np.real(D), np.real(V)
    
        ev_sorted = np.sort(D)
        sort_idx = np.argsort(D)
        sort_idx = sort_idx[::-1]
        ev_sorted = ev_sorted[::-1]
        V = V[:, sort_idx]
        tol = ev_sorted[0] * 10 ** -6
        r = np.sum(ev_sorted > tol)
    
        if r < n_channels:
            lambda2 = ev_sorted[0:r].reshape((1, r))
            M = V[:, 0:r] * (1 / np.sqrt(lambda2))
        else:
            M = np.eye(n_channels)
    
        C_s_r = np.dot(np.dot(M.T, C_s), M)
        C_n_r = np.dot(np.dot(M.T, C_n), M)
        # solve eigenvalue decomposition
        # evals, evecs = linalg.eig(C_s, C_n)
        evals, evecs = linalg.eigh(C_s_r, C_n_r+C_n_r)

        evals = evals.real
        evecs = evecs.real
        # index of sorted eigenvalues
        # ix = np.argsort(np.abs(evals))[::-1]
        ix = np.argsort(evals)[::-1]

        # sort eigenvectors
        W = evecs[:, ix]
        
        W = np.dot(M, W) #filters in columns
        
        
        # spatial patterns
        # A = C * W / (W'* C * W);
        A = linalg.lstsq(np.dot(np.dot(W.T, C_s), W), np.dot(W.T, C_s))[0]
        # A = A.T
        self.patterns_ = A  # n_channels x n_channels (each row is a patter)
        self.filters_ = W.T  # n_channels x n_channels (each row is a filter)
        
        
                
        return self
 

    def best_component(self,q): 
        """
        this is an implementation for finding the best number of components 
        given the power. This is based on [2]
        """
        W=self.filters_       #each row a filter
        Q25=np.percentile(np.var(W, axis=1),25)
        Q75=np.percentile(np.var(W, axis=1),75)

        thr=Q75+q*(Q75-Q25)
        n_comps=np.where(np.var(W, axis=1)>=thr)[0]
        if len(n_comps)==0: 
            #if there is no component reaching the thresholding
            #select at lest the half of the components
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

        if not isinstance(self.n_components, int):
            n_comps=self.best_component(q=self.n_components)
            self.n_components=n_comps
           
        #project data on source space
        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])
        
        if self.denoised:
            #back-project data on signal space
            pick_patterns = self.patterns_[:self.n_components]
            X=np.asarray([np.dot(pick_patterns.T, epoch) for epoch in X])
            if self.return_filtered:
                #filter data
                n_epochs, n_channels, n_samples = X.shape
                signal_band = self.freq[0] #signal bandpass band
                #rephase for filtering
                X_aux=np.reshape(X, [n_epochs, n_channels*n_samples])
                X_s=mne.filter.filter_data(X_aux, self.sampling_freq, l_freq=signal_band[0], h_freq=signal_band[1],method='iir', phase='zero-double')  
                #reshape to original shape
                X=np.reshape(X_s, [n_epochs,n_channels,n_samples])
                
               

        return X
    
 