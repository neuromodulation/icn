import filter
import IO
import projection
import online_analysis
import offline_analysis
import numpy as np
import json
import os
import pickle 

if __name__ == "__main__":

    settings = IO.read_settings()  # reads settings from settings/settings.json file in a dict 
                                   # settings need to be defined for individual runs
                                   # implement settings folder in BIDS? 

    # specify BIDS run 

    vhdr_file = '/Users/hi/Documents/lab_work/BIDS/sub-000/ses-right/eeg/sub-000_ses-right_task-force_run-3_eeg.vhdr'

    #vhdr_files = IO.read_all_vhdr_filenames(settings['BIDS_path'])
    #vhdr_file = vhdr_files[3]
    
    # read grid from session
    cortex_left, cortex_right, subcortex_left, subcortex_right = IO.read_grid()
    grid_ = [cortex_left, subcortex_left, cortex_right, subcortex_right]

    bv_raw, ch_names = IO.read_BIDS_file(vhdr_file)

    subject, run, sess = IO.get_sess_run_subject(vhdr_file)

    sess_right = IO.sess_right(sess)

    # read channels_M1 specifications
    M1_specs = IO.read_M1_channel_specs(vhdr_file[:-9])

    # read channels that are meant to be used in the analysis, currently json file in settings folder
    used_channels = IO.read_used_channels()

    # extract used channels/labels from brainvision file, split up in cortex/subcortex/labels
    dat_cortex, dat_subcortex, dat_label, ind_cortex, ind_subcortex, ind_label, ind_dat = IO.get_dat_cortex_subcortex(bv_raw, ch_names, used_channels)

    # read all used coordinates from session coordinates.tsv BIDS file
    coord_patient = IO.get_patient_coordinates(ch_names, ind_cortex, ind_subcortex, vhdr_file, settings['BIDS_path'])

    # given those coordinates and the provided grid, estimate the projection matrix
    proj_matrix_run = projection.calc_projection_matrix(coord_patient, grid_, sess_right, settings['max_dist_cortex'], settings['max_dist_subcortex'])

    # from the BIDS run channels.tsv read the sampling frequency 
    # here: the sampling frequency can be different for all channels
    # therefore read the frequency for all channels, which makes stuff complicated...
    fs = IO.read_run_sampling_frequency(vhdr_file)

    # read line noise from participants.tsv
    line_noise = IO.read_line_noise(settings['BIDS_path'],subject)

    resample_factor = fs/settings['fs_new']
    seglengths = np.array([fs/1, fs/2, fs/2, fs/2, \
              fs/2, fs/10, fs/10, fs/10]).astype(int)

    normalization_samples = settings['normalization_time']*settings['fs_new']
    new_num_data_points = int((bv_raw.shape[1]/fs)*settings['fs_new'])

    # downsample_idx states the original brainvision sample indexes are used
    downsample_idx = (np.arange(0,new_num_data_points,1)*fs/settings['fs_new']).astype(int)

    filter_fun = filter.calc_band_filters(settings['f_ranges'], fs)

    offset_start = int(seglengths[0] / (fs/settings['fs_new']))

    arr_act_grid_points = IO.get_active_grid_points(sess_right, ind_label, ch_names, proj_matrix_run, grid_)

        #real time analysis
    # for the real time prediction it is necessary to load a previously trained classifier
    real_time_analysis = False 
    if real_time_analysis is True:
        grid_classifiers = np.load('grid_classifiers.npy', allow_pickle=True)  
        estimates = online_analysis.real_time_simulation(fs, settings['fs_new'], seglengths, settings['f_ranges'], grid_, downsample_idx, bv_raw, line_noise, \
                        sess_right, dat_cortex, dat_subcortex, dat_label, ind_cortex, ind_subcortex, ind_label, ind_dat, \
                        filter_fun, proj_matrix_run, arr_act_grid_points, grid_classifiers, normalization_samples, ch_names)


    rf_data_median, pf_data_median, label_median = offline_analysis.preprocessing(fs, settings['fs_new'], seglengths, settings['f_ranges'], grid_, downsample_idx, bv_raw, line_noise, \
                      sess_right, dat_cortex, dat_subcortex, dat_label, ind_cortex, ind_subcortex, ind_label, ind_dat, \
                      filter_fun, proj_matrix_run, arr_act_grid_points, new_num_data_points, ch_names, normalization_samples)

    run_ = {
        "vhdr_file" : vhdr_file,
        "fs_new" : settings['fs_new'],
        "BIDS_path" : settings['BIDS_path'], 
        "projection_grid" : grid_, 
        "bv_raw" : bv_raw, 
        "ch_names" : ch_names, 
        "subject" : subject, 
        "run" : run, 
        "sess" : sess, 
        "sess_right" :  sess_right, 
        "used_channels" : used_channels, 
        "dat_cortex" : dat_cortex, 
        "dat_subcortex" : dat_subcortex, 
        "dat_label" : dat_label, 
        "ind_cortex" : ind_cortex, 
        "ind_subcortex" : ind_subcortex, 
        "ind_label" : ind_label, 
        "ind_label" : ind_dat, 
        "coord_patient" : coord_patient, 
        "proj_matrix_run" : proj_matrix_run, 
        "fs" : fs, 
        "line_noise" : line_noise, 
        "resample_factor" : resample_factor, 
        "seglengths" : seglengths, 
        "normalization_samples" : normalization_samples, 
        "new_num_data_points" : new_num_data_points, 
        "downsample_idx" : downsample_idx, 
        "filter_fun" : filter_fun, 
        "offset_start" : offset_start, 
        "arr_act_grid_points" : arr_act_grid_points, 
        "rf_data_median" : rf_data_median, 
        "pf_data_median" : pf_data_median, 
        "label_median" : label_median
    }

    out_path = os.path.join(settings['out_path'],'sub_' + subject + '_sess_' + sess + '_run_' + run + '.p')
    
    with open(out_path, 'wb') as handle:
        pickle.dump(run_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # we can/should also save the file as json, since this can be also read from MATLAB,
    # here it is necessary though to transform every single numpy array to a list, e.g. np.arr.tolist()
    #json.dump(run_, open(os.path.join(settings['out_path'], vhdr_file, '.json', 'w' )))

    # previously analyzed run:
    #rf_data_median = np.load('rf_data_median.npy')
    #pf_data_median = np.load('pf_data_median.npy')
    #label_con = np.load('dat_con.npy')
    #label_ips = np.load('dat_ips.npy')

