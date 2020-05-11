import filter
import numpy as np 
import projection
import time
from matplotlib import pyplot as plt 

def append_time_dim(X, y_=None, time_stamps=5):
    """
    :param X: in shape(time, grid_points/channels, f_bands)
    apply added time dimension for the data array and label given time_stamps (with downsample_rate=100) in 100ms / need to check with 1375Hz
    """
    if len(X.shape) == 3:
        num_time = X.shape[0]
        num_channels = X.shape[1]
        num_f_bands = X.shape[2]

        time_arr = np.zeros([num_time-time_stamps, num_channels, int(time_stamps*num_f_bands)])
        for ch in range(num_channels):
            for time_idx, time_ in enumerate(np.arange(time_stamps, num_time)):
                for time_point in range(time_stamps):
                    time_arr[time_idx, ch, time_point*num_f_bands:(time_point+1)*num_f_bands] = X[time_-time_point,ch,:]

        if y_ is None:
            return time_arr
        return time_arr, y_[time_stamps:]
    elif len(X.shape) == 2:
        if time_stamps == X.shape[0]:
            time_arr = np.zeros([1+X.shape[0]-time_stamps, int(time_stamps*X.shape[1])])
            #print(time_arr.shape)
            for time_idx, time_ in enumerate(np.arange(time_stamps-1, X.shape[0])):
                #print(time_idx)
                #print('time_:'+str(time_))
                for time_point in range(time_stamps):
                    #print('time_point: '+str(time_point))
                    time_arr[time_idx, time_point*X.shape[1]:(time_point+1)*X.shape[1]] = X[time_-time_point,:]
        else:
            time_arr = np.zeros([X.shape[0]-time_stamps, int(time_stamps*X.shape[1])])
            for time_idx, time_ in enumerate(np.arange(time_stamps, X.shape[0])):
                for time_point in range(time_stamps):
                    time_arr[time_idx, time_point*X.shape[1]:(time_point+1)*X.shape[1]] = X[time_-time_point,:]
        if y_ is None:
            return time_arr
        return time_arr, y_[time_stamps:]
    

def predict(pf_stream, grid_classifiers, arr_act_grid_points):
    res_predict = np.zeros([num_grid_points])
    X = np.clip(pf_stream, -2, 2)
    for grid_point in range(arr_act_grid_points.shape[0]):
        if arr_act_grid_points[grid_point] == 0:
            continue
        
        X_test = X[:,grid_point,:]
        X_test_reshaped = np.reshape(X_test, (X_test.shape[0]*X_test.shape[1]))
        model = grid_classifiers[grid_point]
        res_predict[grid_point] = model.predict(np.expand_dims(X_test_reshaped, axis=0))
    return res_predict

def simulate_data_stream(bv_raw, ind_DAT, ind_time, fs):
    #time.sleep(1/fs)
    return bv_raw[ind_DAT, ind_time]


def real_time_simulation(fs, fs_new, seglengths, f_ranges, grid_, downsample_idx, bv_raw, line_noise, \
                      sess_right, dat_cortex, dat_subcortex, dat_label, ind_cortex, ind_subcortex, ind_label, ind_DAT, \
                      filter_fun, proj_matrix_run, arr_act_grid_points, grid_classifiers, normalization_samples, ch_names):
    
    num_grid_points = grid_[0].shape[1] + grid_[1].shape[1]+ grid_[2].shape[1]+ grid_[3].shape[1]

    label_con = dat_label[1,:][::100][10:]
    label_ips = dat_label[0,:][::100][10:]
    
    dat_buffer = np.zeros([ind_DAT.shape[0], 1000])
    rf_data_rt = np.zeros([ind_DAT.shape[0], len(f_ranges)])
    pf_data_rt = np.zeros([num_grid_points, len(f_ranges)])


    fig = plt.figure(figsize=(10, 5))
    #ax = fig.add_subplot(111)
    #plt.ion()
    #plt.title('label predictions grid point 42')
    #plt.show()
    

    dat_buffer = np.zeros([ind_DAT.shape[0], 1000])
    dat_res = np.zeros([num_grid_points, 100])
    dat_label_con = np.zeros([100])
    dat_label_ips = np.zeros([100])
    rf_data_rt = np.zeros([ind_DAT.shape[0], len(f_ranges)])
    pf_data_rt = np.zeros([num_grid_points, len(f_ranges)])

    pf_stream = []
    rf_stream = []
    pf_stream_median = []
    rf_stream_median = []
    estimates = []
    buffer_counter = 0
    idx_stream = 0
    for ind_time in range(bv_raw.shape[1]):
        if idx_stream == 0:
            if buffer_counter < seglengths[0]-1:
                dat_buffer[:, buffer_counter] = simulate_data_stream(bv_raw, ind_DAT, ind_time, fs)
                buffer_counter += 1 
                continue
        else:
            if buffer_counter < seglengths[7]-1:
                dat_buffer[:,:-1] = dat_buffer[:,1:]
                buffer_offset = seglengths[0] - seglengths[-1] # to have steps of 100 ms
                dat_buffer[:, buffer_counter+buffer_offset] = simulate_data_stream(bv_raw, ind_DAT, ind_time, fs)
                buffer_counter += 1 
                continue
        #plt.imshow(dat_buffer, aspect='auto')
        #plt.title('buffer')
        #plt.show()
        #print(ind_time)
        #print(str(np.round(ind_time*(1/fs),2))+' s')
        buffer_counter = 0    
        
        rf_data_rt = np.zeros([ind_DAT.shape[0], len(f_ranges)])
        pf_data_rt = np.zeros([num_grid_points, len(f_ranges)])
        for ch in ind_DAT:  #  think about using multiprocessing pool to do this simulatenously
            dat_ = dat_buffer[ch,:]
            dat_filt = filter.apply_filter(dat_, sample_rate=fs, filter_fun=filter_fun, line_noise=line_noise, seglengths=seglengths)
            rf_data_rt[ch,:] = dat_filt
        
        #plt.imshow(rf_data_rt.T, aspect='auto')
        #plt.title('raw t-f transformed')
        #plt.show()
        
        #PROJECTION of RF_data to pf_data
        dat_cortex = rf_data_rt[ind_cortex,:]
        dat_subcortex = rf_data_rt[ind_subcortex,:]
        proj_cortex, proj_subcortex = projection.get_projected_cortex_subcortex_data(proj_matrix_run, sess_right, dat_cortex, dat_subcortex)
        pf_data_rt = projection.write_proj_data(ch_names, sess_right, dat_label, ind_label, proj_cortex, proj_subcortex)
        
        #plt.imshow(pf_data_rt.T, aspect='auto')
        #plt.title('projected t-f transformed')
        #plt.show()
        
        if idx_stream<normalization_samples:
            if idx_stream == 0:
                n_idx = 0
            else:
                n_idx = np.arange(0,idx_stream,1)
        else:
            n_idx = np.arange(idx_stream-normalization_samples, idx_stream, 1)
        
        if idx_stream == 0:

            pf_stream.append(pf_data_rt)
            pf_stream_median.append(pf_data_rt)
            
            rf_stream.append(rf_data_rt)
            rf_stream_median.append(rf_data_rt)
        else:
            
            rf_stream.append(rf_data_rt)
            median_ = np.median(np.array(rf_stream)[n_idx,:,:], axis=0)
            rf_stream_val = (rf_data_rt - median_) / median_
            rf_stream_median.append(rf_stream_val)
            
            pf_stream.append(pf_data_rt)
            median_ = np.median(np.array(pf_stream)[n_idx,:,:][:,arr_act_grid_points>0,:], axis=0)
            pf_data_rt_median = (pf_data_rt[arr_act_grid_points>0,:] - median_) / median_
            pf_data_set = np.zeros([num_grid_points, len(f_ranges)])
            pf_data_set[arr_act_grid_points>0,:] = pf_data_rt_median
            pf_stream_median.append(pf_data_set)
            
            #plt.imshow(pf_data_rt.T, aspect='auto')
            #plt.title('projected and resampled t-f transformed')
            #plt.show()
            
            # now use the predictors to estimate the labelement 
            if idx_stream >= 5:
                time_stamp_tf_dat = np.array(pf_stream_median)[-5:,:,:]
                #plt.imshow(time_stamp_tf_dat[:,:,0].T, aspect='auto')
                #plt.clim(-10,10)
                #plt.show()
                predictions = predict(time_stamp_tf_dat, grid_classifiers, arr_act_grid_points)
                estimates.append(predictions)
                
                dat_res[:,:-1] = dat_res[:,1:]
                dat_res[:,-1] = predictions
                
                dat_label_con[:-1] = dat_label_con[1:]
                dat_label_con[-1] = label_con[idx_stream-5]
                
                dat_label_ips[:-1] = dat_label_ips[1:]
                dat_label_ips[-1] = label_ips[idx_stream-5]
                
                
                plt.clf()
                plt.plot(dat_res[46,:], label='prediction', c='green')
                plt.plot(dat_label_con, label='contralateral force', c='red')
                plt.plot(dat_label_ips, label='ipsilateral force', c='blue')
                plt.legend(loc='upper left')
                plt.ylabel('Force')
                plt.xlabel('Time 0.1s')
                plt.ylim(-1, 6)
                if idx_stream == 5:
                    plt.show()
                else:
                    #plt.draw()
                    fig.canvas.draw()
                #fig.canvas.flush_events()
                
                #usage for matplotlib
                '''
                ax.clear()
                ax.plot(dat_res[46,:], label='prediction', c='green')
                ax.plot(dat_label_con, label='contralateral force', c='red')
                ax.plot(dat_label_ips, label='ipsilateral force', c='blue')
                ax.legend(loc='upper left')
                ax.set_ylabel('Force')
                ax.set_xlabel('Time 0.1s')
                ax.set_ylim(-1, 6)
                fig.canvas.draw()
                '''
            
        idx_stream += 1
        
    return estimates