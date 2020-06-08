import numpy as np

def calc_projection_matrix(coord_arr, grid_, sess_right, max_dist_cortex = 20, max_dist_subcortex = 10):
    """
    calculates a projection matrix based on the used coord_arr of that BIDS run and the provided grid
    :param coord_arr: shape: (4) - cortex LEFT; subcortex_LEFT; cortex RIGHT; subcortex_RIGHT coordinate channel grid
    :param grid_: list with cortex left, subcortex left, cortex right, subcortex right coordinate grids
    :param max_dist_cortex: float - defines interpolation parameter, for a given grid point take all 
        channels into account that have a euclidean distance to that channel in the max_dist_cortex range
    :param max_dist_subcortex: float - defines interpolation parameter, for a given grid point take all 
        channels into account that have a euclidean distance to that channel in the max_dist_subcortex range
    :param sess_right - boolean - determines if electrodes had been recorded from left or right hemisphere
    :return: projection matrix array in shape 4: cortex LEFT; subcortex_LEFT; cortex RIGHT; subcortex_RIGHT
        here for each cortex/subcortex LEFT/RIGHT location, the output has shape (grid_point X channel_in_location)
        for one grid point, the sum of all channel coefficients sums up to 1 
    """

    proj_matrix_run = np.empty(2, dtype=object)
    
    if sess_right is True: 
        grid_session = grid_[2:]
    else:
        grid_session = grid_[:2]
    
    
    for loc_, grid in enumerate(grid_session):
        
        if loc_ == 0:   # cortex
            max_dist = max_dist_cortex
        elif loc_ == 1:  # subcortex
            max_dist = max_dist_subcortex
            
        if coord_arr[loc_] is None:  #this checks if there are cortex/subcortex channels in that run
            continue

        channels = coord_arr[loc_].shape[0]
        distance_matrix = np.zeros([grid.shape[1], channels])

        for project_point in range(grid.shape[1]):
            for channel in range(coord_arr[loc_].shape[0]):
                distance_matrix[project_point, channel] = \
                    np.linalg.norm(grid[:, project_point] - coord_arr[loc_][channel, :])
        
        
        proj_matrix = np.zeros(distance_matrix.shape)
        for grid_point in range(distance_matrix.shape[0]):
            used_channels = np.where(distance_matrix[grid_point, :] < max_dist)[0]

            rec_distances = distance_matrix[grid_point, used_channels]
            sum_distances = np.sum(1 / rec_distances)

            for ch_idx, used_channel in enumerate(used_channels):
                proj_matrix[grid_point, used_channel] = (1 / distance_matrix[grid_point, used_channel]) / sum_distances
        proj_matrix_run[loc_] = proj_matrix
        
    return proj_matrix_run

def get_projected_cortex_subcortex_data(proj_matrix_run, sess_right, dat_cortex=None, dat_subcortex=None):
    """
    :param proj_matrix_run - nparray that defines in shape (grid_points X channels) the projection weights
    :param sess_right - boolean - states if the session is left or right 
    :param dat_cortex - nparray - of cortex to project to grid 
    :param dat_subcortex - nparray - of STM to project to grid 
    :return projection cortex data, projected subcortex data
    """
    proj_cortex = None
    proj_subcortex = None

    if dat_cortex is not None:
        proj_cortex = proj_matrix_run[0] @ dat_cortex
    if dat_subcortex is not None:
        proj_subcortex = proj_matrix_run[1] @ dat_subcortex

    return proj_cortex, proj_subcortex


def write_proj_data(ch_names, sess_right, dat_label, ind_label, grid_, proj_cortex=None, proj_subcortex=None):
    """
    :param proj_cortex - projected data on cortex grid 
    """
    num_grid_points = grid_[0].shape[1] + grid_[1].shape[1] + grid_[2].shape[1] + grid_[3].shape[1]

    if proj_cortex is not None:
        arr_all = np.empty([num_grid_points, proj_cortex.shape[1]])
    else:
        arr_all = np.empty([num_grid_points, proj_subcortex.shape[1]])
        
    mov_channel = np.array(ch_names)[ind_label]

    Con_label = False; Ips_label = False
    
    #78 = grid_[0].shape[1] + grid_[2].shape[1]
    #86 = grid_[0].shape[1] + grid_[1].shape[1] + grid_[2].shape[1]
    

     #WRITE CONTRALATERAL DATA if the respective movement channel exists
    if sess_right is True:
        if len([ch for ch in mov_channel if 'LEFT' in ch]) >0:
            Con_label =True
            dat_label_con = dat_label[[ch_idx for ch_idx, ch in enumerate(mov_channel) if 'LEFT' in ch][0],:]
            arr_all[:grid_[0].shape[1],:] = proj_cortex
            if proj_subcortex is not None:
                 arr_all[(grid_[0].shape[1] + grid_[2].shape[1]):(grid_[0].shape[1] + grid_[1].shape[1] + grid_[2].shape[1]),:] = proj_subcortex

    elif sess_right is False:
        if len([ch for ch in mov_channel if 'RIGHT' in ch]) >0:
            Con_label =True
            dat_label_con = dat_label[[ch_idx for ch_idx, ch in enumerate(mov_channel) if 'RIGHT' in ch][0],:]
            arr_all[:grid_[0].shape[1],:] = proj_cortex
            if proj_subcortex is not None:
                 arr_all[grid_[0].shape[1] + grid_[2].shape[1]:grid_[0].shape[1] + grid_[1].shape[1] + grid_[2].shape[1],:] = proj_subcortex


    #WRITE IPSILATERAL DATA if the respective movement channel exists
    if sess_right is False:
        if len([ch for ch in mov_channel if 'LEFT' in ch]) >0:
            Ips_label = True
            dat_label_ips = dat_label[[ch_idx for ch_idx, ch in enumerate(mov_channel) if 'LEFT' in ch][0],:]

            arr_all[grid_[0].shape[1]:grid_[0].shape[1]+grid_[2].shape[1],:] = proj_cortex
            if proj_subcortex is not None:
                 arr_all[grid_[0].shape[1] + grid_[1].shape[1] + grid_[2].shape[1]:,:] = proj_subcortex
    elif sess_right is True:
        if len([ch for ch in mov_channel if 'RIGHT' in ch]) >0:
            Ips_label = True
            dat_label_ips = dat_label[[ch_idx for ch_idx, ch in enumerate(mov_channel) if 'RIGHT' in ch][0],:]
            arr_all[grid_[0].shape[1]:grid_[0].shape[1]+grid_[2].shape[1],:] = proj_cortex
            if proj_subcortex is not None:
                 arr_all[grid_[0].shape[1]+grid_[1].shape[1]+grid_[2].shape[1]:,:] = proj_subcortex
    #ind_active_ = np.where(np.sum(arr_all, axis=1) != 0)
    return arr_all