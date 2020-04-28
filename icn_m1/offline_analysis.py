import filter
import numpy as np
import projection
from scipy import sparse
from scipy.sparse.linalg import spsolve
import cvxpy as cp
from scipy import signal

#%%

def preprocessing(fs, fs_new, seglengths, f_ranges, grid_, downsample_idx, bv_raw, line_noise, \
                      sess_right, dat_cortex, dat_subcortex, dat_label, ind_cortex, ind_subcortex, ind_label, ind_DAT, \
                      filter_fun, proj_matrix_run, arr_act_grid_points, new_num_data_points, ch_names, normalization_samples):
    offset_start = int(seglengths[0] / (fs/fs_new))  # offset start is here the number of samples new_fs to skip 
    num_channels = ind_DAT.shape[0]
    num_grid_points = np.concatenate(grid_, axis=1).shape[1] # since grid_ is setup in cortex left, subcortex left, cortex right, subcortex right
    num_f_bands = len(f_ranges)

    rf_data = np.zeros([new_num_data_points-offset_start, num_channels, num_f_bands])  # raw frequency array
    rf_data_median = np.zeros([new_num_data_points-offset_start, num_channels, num_f_bands])
    pf_data = np.zeros([new_num_data_points-offset_start, num_grid_points, num_f_bands])  # projected 
    pf_data_median = np.zeros([new_num_data_points-offset_start, num_grid_points, num_f_bands])  # projected 
    label_median = np.zeros([new_num_data_points, ind_label.shape[0]])
    new_idx = 0

    for c in range(downsample_idx.shape[0]):  
        print(str(np.round(c*(1/fs_new),2))+' s')
        if downsample_idx[c]<seglengths[0]:  # neccessary since downsample_idx starts with 0, wait till 1s for theta is over
            continue

        for ch in ind_DAT:    
            dat_ = bv_raw[ch, downsample_idx[c-offset_start]:downsample_idx[c]]
            dat_filt = filter.apply_filter(dat_, sample_rate=fs, filter_fun=filter_fun, line_noise=line_noise, seglengths=seglengths)
            rf_data[new_idx,ch,:] = dat_filt

        #PROJECTION of RF_data to pf_data
        dat_cortex = rf_data[new_idx, ind_cortex,:]
        dat_subcortex = rf_data[new_idx, ind_subcortex,:]
        proj_cortex, proj_subcortex = projection.get_projected_cortex_subcortex_data(proj_matrix_run, sess_right, dat_cortex, dat_subcortex)
        pf_data[new_idx,:,:] = projection.write_proj_data(ch_names, sess_right, dat_label, ind_label, proj_cortex, proj_subcortex)

        #normalize acc. to Median of previous normalization samples
        if c<normalization_samples:
            if new_idx == 0:
                n_idx = 0
            else:
                n_idx = np.arange(0,new_idx,1)
        else:
            n_idx = np.arange(new_idx-normalization_samples, new_idx, 1)

        if new_idx == 0:
            rf_data_median[n_idx,:,:] = rf_data[n_idx,:,:]
            pf_data_median[n_idx,:,:] = pf_data[n_idx,:,:]
            label_median[n_idx,:] = dat_label[:,n_idx]
        else:
            median_ = np.median(rf_data[n_idx,:,:], axis=0)
            rf_data_median[new_idx,:,:] = (rf_data[new_idx,:,:] - median_) / median_
            
            median_ = np.median(pf_data[n_idx,:,:][:,arr_act_grid_points>0,:], axis=0)
            pf_data_median[new_idx,arr_act_grid_points>0,:] = (pf_data[new_idx,arr_act_grid_points>0,:] - median_) / median_
            
            median_ = np.median(dat_label[:,n_idx], axis=1)
            label_median[new_idx,:] = (dat_label[:,downsample_idx[c]] - median_) / median_
        new_idx += 1
    return rf_data_median, pf_data_median, label_median

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def baseline_als(y, lam, p, niter=10): 
    """
    Baseline drift correction based on [1]
    Inputs: 
       y: row signal to be cleaned (array, numpy array)
       lam: reg. parameter (int)
       p: asymmetric parameter. Value in (0 1). 
       
    Problem to Solve (W + lam*D'*D)z=Wy, 
    where W=diag(w), D=second order diff. matrix (linear problem)
    [1] P. H. C. Eilers, H. F. M. Boelens, Baseline correction with asymmetric least squares smoothing, 
    Leiden University Medical Centre report, 2005.
    """ 
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def baseline_rope(y, lam=1):
    """
   Baseline drift correction based on [1]
   Inputs: 
       y: row signal to be cleaned (array, numpy array)
       lam: reg. parameter (int)

   Problem to Solve min |y-b| + lam*(diff_b)^2, s.t. b<=y 
   
   [1] Xie, Z., Schwartz, O., & Prasad, A. (2018). Decoding of finger 
   trajectory from ECoG using deep learning. Journal of neural engineering,
   15(3), 036009.
  """ 
    b = cp.Variable(y.shape)  
    objective = cp.Minimize(cp.norm(y-b, 2)+lam*cp.sum_squares(cp.diff(b, 1)))
    constraints = [b <= y]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="SCS")
    z=b.value #--> baseline
              
    return z

def baseline_correction(y, method='baseline_als', param=[1e2, 1e-4], thr=1e-1, normalize=True, Decimate=1, Verbose=True):
    """
    
    Parameters
    ----------
    y : array/np.array
        raw signal to be corrected
    method : string, optional
        two possible method for baseline correction are allowed 'baseline_rope'
        and 'baseline_als'. See documentation of each method. The default is 'baseline_rope'.
    param : number or array of numbers, optional
        parameters needed in each optimization method. If baseline_rope is being used, param 
        refers to the regularization parameter. If baseline_als is being used
        param should be a 2-lenght array which first value is the regularization
        parameter and the second is the weigthed value. The default is [1e2, 1e-4].
    thr : number, optional
        threshold value in each small variation between trails could still remains
        after baseline elimination. The default is 1e-1.
    normalize : boolean, optional
        if normalize is True the original signal as well as the output corrected signal
        will be scalled between 0 and 1. The default is True.
    Decimate: number, optinal
        before baseline correction it might be necessary to downsample the original raw
        signal. We recommend to do this step. The default is 1, i.e. no decimation.
    Verbose: boolean, optional
        The default is True.

    Returns
    -------
    y_corrected: signal with baseline correction
    onoff: squared signal useful for onset target evaluation.
    y: original signal


    """
    if Decimate!=1:
        if Verbose: print('>>Signal decimation is being done')
        y=signal.decimate(y, Decimate)

    if method=='baseline_als' and np.size(param)!=2:
        raise ValueError("If baseline_als method is desired, param should be a 2 length object")
    if method=='baseline_rope' and np.size(param)>1:
        raise ValueError("If baseline_rope method is desired, param should be a number")
        
    if method=='baseline_als':
        if Verbose: print('>>baseline_als is being used')
        z=baseline_als(y, lam=param[0], p=param[1])
    else: 
        if Verbose: print('>>baseline_rope is being used')
        z=baseline_rope(y, lam=param)  
        
    #subtract baseline
    y_corrected=y-z
    
    #normalize
    if normalize:
        y=NormalizeData(y)
        y_corrected=NormalizeData(y_corrected)
        
    #eliminate interferation
    y_corrected[y_corrected<thr]=0      
    #create on-off signal
    onoff=np.zeros(np.size(y_corrected))
    onoff[y_corrected>0]=1
    return y_corrected, onoff, y

