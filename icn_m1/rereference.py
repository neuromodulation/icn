import numpy as np
import pandas as pd
import IO

def rereference(run_string, data_cortex=None, data_subcortex=None):
    """
    This function rereference data to the indicated channel reference in the files
    "*_channels_MI.tsv". This file must be customized by the user before running this 
    script.

    Parameters
    ----------
    run_string (string): run string without specific ending in form sub-000_ses-right_task-force_run-0

    data_cortex : array, shape(n_channels, n_samples)
        raw data cortex to be epoched. Default it None.
    data_subcortex : array, shape(n_channels, n_samples)
        raw data subcortex to be epoched. Default it None.

    
    Returns
    -------
    new_raw_data : array, shape(n_channels, n_samples)
        rereferenced raw_data.

    """
    
    df_channel = pd.read_csv(run_string + "channels_M1.tsv", sep="\t")
    #non_target_ch = np.where(df_channel['target'] == 0)[0]
    
    used_channels = IO.read_M1_channel_specs(run_string)

    #index_channels = np.intersect1d(used_ch['subcortex'], non_target_ch)
    
    channels_name=df_channel['name'].tolist()
    
    if data_subcortex is not None: 
        index_channels=used_channels['subcortex']-min(used_channels['subcortex'])
        new_data_subcortex=data_subcortex.copy()
           
        for i in index_channels:
            
            elec_channel=index_channels==i
            ch=data_subcortex[elec_channel,:]
            if df_channel['rereference'][i] == 'average':
                av=np.mean(data_subcortex[index_channels!=i,:], axis=0)
                new_data_subcortex[i]=ch-av
            else:
                index=[]
                ref_channels=df_channel['rereference'][i].split('+')
    
                for j in range(len(ref_channels)):
                    if ref_channels[j] not in channels_name:
                        raise ValueError('One or maybe more of the ref_channels are not part of the recording channels.')
                    index.append(channels_name.index(ref_channels[j]))
                    
                new_data_subcortex[i]=ch-np.mean(data_subcortex[index,:], axis=0)
    else:
        new_data_subcortex=None
    
    if data_cortex is not None:
        index_channels=used_channels['cortex']-min(used_channels['cortex'])

        new_data_cortex=data_cortex.copy()
        
        for i in index_channels:
        
            elec_channel=index_channels==i
            ch=data_cortex[elec_channel,:]
            if df_channel['rereference'][i] == 'average':
                av=np.mean(data_cortex[index_channels!=i,:], axis=0)
                new_data_cortex[i]=ch-av
            else:
                index=[]
                ref_channels=df_channel['rereference'][i].split('+')
    
                for j in range(len(ref_channels)):
                    if ref_channels[j] not in channels_name:
                        raise ValueError('One or maybe more of the ref_channels are not part of the recording channels.')
                    if ref_channels[j]==channels_name[i]:
                        raise ValueError('You cannot rereference to the same channel.')
                    index.append(channels_name.index(ref_channels[j]))
                    
                new_data_cortex[i]=ch-np.mean(data_cortex[index,:], axis=0)
    else:
        new_data_cortex=None
            
    return new_data_cortex, new_data_subcortex
