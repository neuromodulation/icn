import filter
import IO
import sys
import numpy as np
import offline_analysis
import time
import json

#given the ieeg raw file write a generator
def ieeg_raw_generator(ieeg_raw, fs, fs_new, offset_start):
     
    cnt_fsnew = 0
    for cnt in range(ieeg_raw.shape[1]):
        if cnt < offset_start:
            cnt_fsnew +=1
            continue
        
        cnt_fsnew +=1
        if cnt_fsnew >= (fs/fs_new):
            cnt_fsnew = 0
            yield ieeg_raw[:,cnt-offset_start:cnt]

with open('settings\\settings.json', 'rb') as f:
        settings = json.load(f)

ieeg_files = IO.get_all_ieeg_files(settings['BIDS_path'])

subject, run, sess = IO.get_sess_run_subject(ieeg_files[0])

fs=IO.read_run_sampling_frequency(ieeg_files[0])[0]

fs = int(np.ceil(fs))

line_noise = IO.read_line_noise(settings['BIDS_path'],"002") # the line noise column is missing in the 

filter_len = fs 
filter_fun = filter.calc_band_filters(settings['frequencyranges'], sample_rate=fs, filter_len=filter_len)

ieeg_raw, ch_names = IO.read_BIDS_file(ieeg_files[0])

offset_start = int(np.ceil(settings["seglengths"][0] * fs))
gen_ = ieeg_raw_generator(ieeg_raw, fs=fs, fs_new=settings["resamplingrate"], offset_start=offset_start)

time_start = time.time()
res = run(gen_, settings["seglengths"], settings["f_ranges"], line_noise, fs, settings["resamplingrate"], \
    filter_fun, num_channels=ieeg_raw.shape[0], clip_low=-2, clip_high=2, \
    usemean_=True, normalize=True, normalize_time=settings["normalization_time"])
time_stop = time.time()
print(time_stop -time_start)

# save object
# call plotting functions
