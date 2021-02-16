import numpy as np

def get_hjorth_raw(features_, data_, ch):
    
    features_[ch+"_RawHjorth_Activity"] = np.var(data_)
    deriv_variance = np.var(np.diff(data_))
    mobility = np.sqrt(deriv_variance / np.var(data_))
    features_[ch+"_RawHjorth_Mobility"] = mobility

    dat_deriv_2_var = np.var(np.diff(np.diff(data_)))
    deriv_mobility = np.sqrt(dat_deriv_2_var / np.var(np.diff(data_)))
    features_[ch+"_RawHjorth_Complexity"] = deriv_mobility / mobility

    return features_