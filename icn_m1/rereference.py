import numpy as np
import pandas as pd

# def rereference(bv_raw, df_channel):
#     # given used and predictor column, rereference those that have used = 1 and predictor = 0 acc to CAR and BR 

#     bv_raw_r = np.zeros(bv_raw.shape)

#     used_ch = np.where(df_channel['used'] == 1)[0]

#     non_predictor_ch = np.where(df_channel['predictor'] == 0)[0]

#     ch_ = np.intersect1d(used_ch, non_predictor_ch)

#     for ch in ch_:
#         if df_channel['rereference'] == 'CAR':
#             avg_ = np.mean(bv_raw[np.where(df_channel['rereference'] == 'CAR')[0],:], axis=0)  # take the avg of all CAR channels and subtract it 
#             bv_raw_r[ch,:] -= avg_
#         elif:

def rereference(raw_data, run_string):
    """
    

    Parameters
    ----------
    raw_dat : array, shape(n_channels, n_samples)
        raw_data to be epoched.
    run_string (string): run string without specific ending in form sub-000_ses-right_task-force_run-0

    
    Returns
    -------
    new_raw_data : array, shape(n_channels, n_samples)
        rereferenced raw_data.

    """
    
    df_channel = pd.read_csv(run_string + "_channels_M1.tsv", sep="\t")
    used_ch = np.where(df_channel['used'] == 1)[0]
    non_target_ch = np.where(df_channel['target'] == 0)[0]

    index_channels = np.intersect1d(used_ch, non_target_ch)

    new_raw_data=raw_data.copy()
   
    channels_name=df_channel['name'].tolist()
    for i in index_channels:
        #print(df_channel['name'][i])
        
        elec_channel=index_channels==i
        ch=raw_data[elec_channel,:]
        if df_channel['rereference'][i] == 'average':
            av=np.mean(raw_data[index_channels!=i,:], axis=0)
            new_raw_data[i]=ch-av
        else:
            index=[]
            ref_channels=df_channel['rereference'][i].split('+')

            for j in range(len(ref_channels)):
                if ref_channels[j] not in channels_name:
                    raise ValueError('One or maybe more of the ref_channels are not part of the recording channels.')
                index.append(channels_name.index(ref_channels[j]))
                
            new_raw_data[i]=ch-np.mean(raw_data[index,:], axis=0)
            
    return new_raw_data
