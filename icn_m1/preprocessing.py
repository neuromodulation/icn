import numpy as np
import pandas as pd
import IO

def rereference(run_string, bv_raw, get_ch_names=True, get_cortex_subcortex=False):
    """
    This function rereference data to the indicated channel reference in the files
    "*_channels_MI.tsv". This file must be customized by the user before running this 
    script.

    Parameters
    ----------
    run_string (string): run string without specific ending in form sub-000_ses-right_task-force_run-0

    bv_raw : array, shape(n_channels, n_samples)
        raw data 

    
    Returns
    -------
    new_data : array, shape(n_channels, n_samples)
        rereferenced raw_data.

    """
    
    df_channel = pd.read_csv(run_string + "_channels_M1.tsv", sep="\t")
    #non_target_ch = np.where(df_channel['target'] == 0)[0]
    
    used_channels = IO.read_M1_channel_specs(run_string)

   
    channels_name=df_channel['name'].tolist()
    
    new_data=bv_raw.copy()

    if used_channels['subcortex'] is not None: 
        data_subcortex = bv_raw[used_channels['subcortex'],:]
        index_channels=used_channels['subcortex']
        new_data_subcortex=data_subcortex.copy()
        idx=0
        for i in index_channels:
            
            elec_channel=index_channels==i
            ch=data_subcortex[elec_channel,:]
            if df_channel['rereference'][i] == '-':
                continue
            if df_channel['rereference'][i] == 'average':
                av=np.mean(data_subcortex[index_channels!=i,:], axis=0)
                new_data_subcortex[idx]=ch-av
            else:
                index=[]
                ref_channels=df_channel['rereference'][i].split('+')
    
                for j in range(len(ref_channels)):
                    if ref_channels[j] not in channels_name:
                        raise ValueError('One or maybe more of the ref_channels are not part of the recording channels.')
                    index.append(channels_name.index(ref_channels[j]))
                    
                new_data_subcortex[idx]=ch-np.mean(data_subcortex[index,:], axis=0)
            idx=idx+1    
        new_data[used_channels['subcortex'],:]=new_data_subcortex 

    else:
        new_data_subcortex=None
    
    if used_channels['cortex'] is not None:
        data_cortex = bv_raw[used_channels['cortex'],:]

        index_channels=used_channels['cortex']
        new_data_cortex=data_cortex.copy()
        idx=0
        for i in index_channels:
       
            elec_channel=index_channels==i
            ch=data_cortex[elec_channel,:]
            if df_channel['rereference'][i] == '-':
                continue
            if df_channel['rereference'][i] == 'average':
                av=np.mean(data_cortex[index_channels!=i,:], axis=0)
                new_data_cortex[idx]=ch-av
              

            else:
                index=[]
                ref_channels=df_channel['rereference'][i].split('+')
    
                for j in range(len(ref_channels)):
                    if ref_channels[j] not in channels_name:
                        raise ValueError('One or maybe more of the ref_channels are not part of the recording channels.')
                    if ref_channels[j]==channels_name[i]:
                        raise ValueError('You cannot rereference to the same channel.')
                    index.append(channels_name.index(ref_channels[j]))
                    
                new_data_cortex[idx]=ch-np.mean(data_cortex[index,:], axis=0)
                
            idx=idx+1
        new_data[used_channels['cortex'],:]=new_data_cortex 
    else:
        new_data_cortex=None
    
    if get_cortex_subcortex:
        if get_ch_names:
            return new_data_cortex, new_data_subcortex, channels_name
        else:
            return new_data_cortex, new_data_subcortex
    else:
        if get_ch_names:
            return new_data, channels_name
        else:
            return new_data
