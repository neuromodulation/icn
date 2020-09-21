import os
import scipy
import numpy as np
from matplotlib import pyplot as plt
from mne import io
from bids import BIDSLayout
from mne.decoding import TimeFrequency
from matplotlib import pyplot as plt
from scipy import stats, signal
import mne
from mne import create_info, EpochsArray
from mne.time_frequency import tfr_morlet
import pandas as pd
import multiprocessing

PATH = "C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\PD_ButtonPress\\"
patient_files = ["sub-FOG006\\ses-postimp\\ieeg\\sub-FOG006_ses-postimp_task-buttonPress_run-01_ieeg.edf",
                 "sub-FOG008\\ses-postimp\\ieeg\\sub-FOG008_ses-postimp_task-buttonPress_run-01_ieeg.edf",
                 "sub-FOG010\\ses-postimp\\ieeg\\sub-FOG010_ses-postimp_task-buttonPress_run-01_ieeg.edf",
                 "sub-FOG011\\ses-postimp\\ieeg\\sub-FOG011_ses-postimp_task-buttonPress_run-01_ieeg.edf",
                 "sub-FOG013\\ses-postimp\\ieeg\\sub-FOG013_ses-postimp_task-buttonPress_run-01_ieeg.edf",
                 "sub-FOGC001\\ses-postimp\\ieeg\\sub-FOGC001_ses-postimp_task-buttonPress_run-01_ieeg.edf"]
subjects = ["FOG006", "FOG008", "FOG010", "FOG011", "FOG013", "FOGC001"]

def calc_epochs(bv_raw, y_tr, info, threshold, epoch_lim):
    ind_mov = np.where(np.diff(np.array(y_tr>threshold)*1) == 1)[0]
    low_limit = ind_mov>epoch_lim
    up_limit = ind_mov < y_tr.shape[0]-epoch_lim
    ind_mov = ind_mov[low_limit & up_limit]
    bv_epoch = np.zeros([ind_mov.shape[0], int(epoch_lim*2)])
    y_arr = np.zeros([ind_mov.shape[0],int(epoch_lim*2)])
    n_epochs = bv_epoch.shape[0]
    events = np.empty((n_epochs, 3), dtype=int)


    event_id = dict(mov_present=1)

    for idx, i in enumerate(ind_mov):
        #if i > 0 and (ind_mov[idx] - ind_mov[idx-1] > 1000):
        # TTL signal is a rectangular signal with length ~500ms
        bv_epoch[idx,:] = bv_raw[i-epoch_lim:i+epoch_lim]
        y_arr[idx,:] = y_tr[i-epoch_lim:i+epoch_lim]
        events[idx,:] = i, 0, event_id["mov_present"]
    print(bv_epoch.shape)
    bv_epoch = np.expand_dims(bv_epoch, axis=1)
    print(bv_epoch.shape)
    epochs = EpochsArray(data=bv_epoch, info=info, events=events, event_id=event_id)
    return epochs

def preprocess_mov(mov_dat):
    # the TIME OFF in the TTL signal is ~50 ms
    MOV_ON = False
    mov_new = np.zeros(mov_dat.shape[0])
    mov_new[0] = mov_dat[0]
    mov_on_set = 0
    for i in range(mov_dat.shape[0]):
        if i > 0 and mov_dat[i] > 1:
            MOV_ON = True
            mov_on_set = i
        if (i - mov_on_set) > 100 and mov_dat[i] < 1:
            mov_new[i] = 0
            MOV_ON = False
        if MOV_ON is True:
            mov_new[i] = 1
    return mov_new

def write_TF(Name_begin, patient_f, patient_idx):
    channel_df = pd.read_csv(os.path.join(PATH, patient_f)[:-9]+"_channels.tsv", sep="\t",
                                 encoding= 'unicode_escape')

    dat = io.read_raw_edf(os.path.join(PATH, patient_f))
    ttl_idx = [idx for idx, ch in enumerate(dat.ch_names) if "POL DC10" in ch][0]
    mov_bin = preprocess_mov(dat.get_data()[ttl_idx,:])

    idx_ecog = []
    for ch in channel_df["name"]:
        if ch.startswith(Name_begin) is False or \
                channel_df.iloc[channel_df[channel_df.name == ch].index[0]]["status"] != "good":
            continue
        idx_ecog = [idx for idx, ch_ in enumerate(dat.ch_names) if ch_ == ch][0]
        info = create_info(ch_names=[ch[4:]], sfreq=2000, ch_types='ecog')
        epochs = calc_epochs(dat.get_data()[idx_ecog,:], mov_bin, info, threshold=0.5, epoch_lim=2500*2)
        freqs = np.arange(7, 200, 1)
        power = tfr_morlet(epochs, freqs=freqs,
                               n_cycles=5, return_itc=False, zero_mean=True, picks=0)
        dat_ = power.data[0,:,1000:9000]  # cut off borders due to Wavelet transform; 500ms till 4s post movement
        dat_z = stats.zscore(dat_, axis=1)
        #plt.imshow(dat_z, aspect='auto', extent=[-2,2,200,0])#, cmap='hot')
        #cbar = plt.colorbar()
        #cbar.set_label('Normalized spectral power [VAR]')
        #plt.clim(-1.5,1.5)
        #plt.gca().invert_yaxis()
        #plt.title(Name_begin[4:] + " " + subjects[patient_idx])
        #plt.show()
        np.save('C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\Shared Lab Folders\\CRCNS\\PD_ButtonPress\\derivatives\\'+
                        "sub_"+subjects[patient_idx] +"_ch_"+ch[4:]+'.npy', dat_z)

if __name__ == '__main__':

    args = []
    for Name_begin in ["POL RS", "POL LD", "POL RD"]:
        for patient_idx, patient_f in enumerate(patient_files):
            args.append([Name_begin, patient_f, patient_idx])

    NUM_PROCESSES = 59
    pool = multiprocessing.Pool(NUM_PROCESSES)
    pool.starmap(write_TF, args)
