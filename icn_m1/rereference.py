import numpy as np

def rereference(bv_raw, df_channel):
    # given used and predictor column, rereference those that have used = 1 and predictor = 0 acc to CAR and BR 

    bv_raw_r = np.zeros(bv_raw.shape)

    used_ch = np.where(df_channel['used'] == 1)[0]

    non_predictor_ch = np.where(df_channel['predictor'] == 0)[0]

    ch_ = np.intersect1d(used_ch, non_predictor_ch)

    for ch in ch_:
        if df_channel['rereference'] == 'CAR':
            avg_ = np.mean(bv_raw[np.where(df_channel['rereference'] == 'CAR')[0],:], axis=0)  # take the avg of all CAR channels and subtract it 
            bv_raw_r[ch,:] -= avg_
        elif:

