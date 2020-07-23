#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:55:45 2020

@author: victoria
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy


class TimeLagFilterBank(BaseEstimator, TransformerMixin):
    """Apply a given indentical pipeline over a bank of filter.
    The pipeline provided with the constrictor will be appield on the 4th
    axis of the input data. This pipeline should be used with a FilterBank
    paradigm.
    the last column of the data in the 3rd axis is the target variable
    This can be used to build a filterbank CSP, for example::
        pipeline = make_pipeline(FilterBank(estimator=CSP()), LDA())
    Parameters
    ----------
    estimator: sklean Estimator
        the sklearn pipeline to apply on each band of the filter bank. e.g: CSP, SPOC
    flatten: bool (True)
        If True, output of each band are concatenated together on the feature
        axis. if False, output are stacked.
   time_stamps: int 
       an integrer indicating of many time windows will be contatenated along features
        
     //
     This function is based on the Filterbank implementation used within
    the MOABB [1] repo available at http://moabb.neurotechx.com/docs/index.html
    
    [1] Jayaram, V., & Barachant, A. (2018). MOABB: trustworthy algorithm 
    benchmarking for BCIs. Journal of neural engineering, 15(6), 066011.
    //
    
    
    
    """

    def __init__(self, estimator, flatten=True, time_stamps=5):
        self.estimator = estimator
        self.flatten = flatten
        self.time_stamps= time_stamps

    def fit(self, X, y=None):
        assert X.ndim == 4
        target=X[:,0,-1,0] #the target is equal at any freq. band or electrode
        data=X[:,:,:-1,:]
        self.models = [
            deepcopy(self.estimator).fit(data[...,i], target)
            for i in range(data.shape[-1])
        ]
        
        self.filters= [self.models[i].filters_ for i in range(data.shape[-1])]
        self.patterns= [self.models[i].patterns_ for i in range(data.shape[-1])]
        
        self.mean=[self.models[i].mean_ for i in range(data.shape[-1])]
        self.std=[self.models[i].std_ for i in range(data.shape[-1])]


        return self
    
    def append_time_dim(self, arr, time_stamps):
        """
        apply added time dimension for the data array and label given time_stamps (with downsample_rate=100) in 100ms / need to check with 1375Hz
        """
        time_arr = np.zeros([np.shape(arr)[0], int(time_stamps*np.shape(arr)[1])])
        for time_idx, time_ in enumerate(np.arange(time_stamps, np.shape(arr)[0])):
            for time_point in range(time_stamps):
                time_arr[time_idx+time_stamps, time_point*np.shape(arr)[1]:(time_point+1)*np.shape(arr)[1]] = arr[time_-time_point,:]
           
        return time_arr
 
   
               

    def transform(self, X):
        assert X.ndim == 4
        data=X[:,:,:-1,:]
        out = [self.models[i].transform(data[...,i]) for i in range(data.shape[-1])]
        assert out[0].ndim == 2, ("Each band must return a n dimensional "
                                  f" matrix, currently have {out[0].ndim}")
        if self.flatten:
            out=np.concatenate(out, axis=1)
            if self.time_stamps>1:
                return self.append_time_dim(out, self.time_stamps)
            else:
                return out
           
        else:
            return np.stack(out, axis=2)

    def __repr__(self):
        estimator_name = type(self).__name__
        estimator_prms = self.estimator.get_params()
        return '{}(estimator={}, flatten={})'.format(
            estimator_name, estimator_prms, self.flatten)
    
 