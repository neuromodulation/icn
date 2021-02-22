import numpy as np 
from numba import jit

#@jit
def get_bandpower_features(features_, s, seglengths, dat_filtered, KF_dict, ch, ch_idx):

    for f_band in range(len(s["bandpass_filter_settings"]["frequency_ranges"])):
        seglength = min((seglengths[f_band], dat_filtered.shape[-1]))
        for bp_feature in s["bandpass_filter_settings"]["bandpower_features"]:
            used = False
            if bp_feature == "activity" and s["bandpass_filter_settings"]["bandpower_features"][bp_feature] is True:
                feature_calc = np.var(dat_filtered[ch_idx, f_band, -seglength:])
                used = True                                
            elif bp_feature == "mobility" and s["bandpass_filter_settings"]["bandpower_features"][bp_feature] is True:
                deriv_variance = np.var(np.diff(dat_filtered[ch_idx, f_band, -seglength:]))
                feature_calc = np.sqrt(deriv_variance / np.var(dat_filtered[ch_idx, f_band, -seglength:]))
                used = True
            elif bp_feature == "complexity" and s["bandpass_filter_settings"]["bandpower_features"][bp_feature] is True:
                dat_deriv = np.diff(dat_filtered[ch_idx, f_band, -seglength:])
                deriv_variance = np.var(dat_deriv)
                mobility = np.sqrt(deriv_variance / np.var(dat_filtered[ch_idx, f_band, -seglength:]))
                dat_deriv_2 = np.diff(dat_deriv)
                dat_deriv_2_var = np.var(dat_deriv_2)
                deriv_mobility = np.sqrt(dat_deriv_2_var / deriv_variance)
                feature_calc = deriv_mobility / mobility
                used = True
            if s["kalman_filter_settings"]["frequency_bands"][f_band] is True and s["methods"]["kalman_filter"] is True:
                KF_dict[ch+"_"+bp_feature+"_"+s["bandpass_filter_settings"]["feature_labels"][f_band]].predict()
                KF_dict[ch+"_"+bp_feature+"_"+s["bandpass_filter_settings"]["feature_labels"][f_band]].update(feature_calc)
                feature_calc = KF_dict[ch+"_"+bp_feature+"_"+s["bandpass_filter_settings"]["feature_labels"][f_band]].x[0] # filtered sensor signal
            if used == True:
                feature_name = ch+"_bandpass_"+bp_feature+"_"+s["bandpass_filter_settings"]["feature_labels"][f_band]
                features_[feature_name] = feature_calc
    return features_
