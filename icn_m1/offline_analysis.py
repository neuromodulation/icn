import filter
import numpy as np
import projection

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