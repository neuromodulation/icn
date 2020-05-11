import numpy as np
import pandas as pd

def rereference(raw_data, run_string):
    """
    This function rereference data to the indicated channel reference in the files
    "*_channels_MI.tsv". This file must be customized by the user before running this 
    script.

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
