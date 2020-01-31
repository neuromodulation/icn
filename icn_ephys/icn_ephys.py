import csv
import os
import pathlib
from collections import OrderedDict
from datetime import datetime, timezone, timedelta

import mne
import numpy as np
import pandas as pd
import pyedflib
from autoreject import (get_rejection_threshold, Ransac)
from matplotlib import pyplot as plt
from mne_bids import write_raw_bids, read_raw_bids, make_bids_basename

import icn_stats as stats
import icn_tb as tb


# internal tools
def _stamp_to_dt(utc_stamp):
    """Convert timestamp to datetime object in Windows-friendly way."""
    # The min on windows is 86400
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return (datetime.fromtimestamp(0, tz=timezone.utc) +
            timedelta(0, stamp[0], stamp[1]))  # day, sec, μs


# toolbox independent functions
def normalize_spectrum(mpow, f):
    rpow = mpow.copy()
    spow = mpow.copy()
    fnorm = np.append(np.arange(f.searchsorted(5), f.searchsorted(45) + 1),
                      np.arange(f.searchsorted(55), f.searchsorted(95) + 1))
    for n in np.arange(0, mpow.shape[0]):
        rpow[n, :] = (mpow[n, :] / mpow[n, fnorm].sum()) * 100
        spow[n, :] = (mpow[n, :] / mpow[n, fnorm].std())
    return rpow, spow


def plot(x, y=None, color=None):
    for a in np.arange(0, y.shape[0]):
        if color is None:
            plt.plot(x, y[a, :])
        elif isinstance(color, type(np.zeros(1))):
            plt.plot(x, y[a, :], color=color[a, :])
        else:
            plt.plot(x, y[a, :], color=color)
    return plt.gca()


def plot_power(f, mpow, channels=None, norm=True):
    if norm:
        pows = normalize_spectrum(mpow, f)
        mpow = pows[0]
    tb.plot(f, mpow)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Spectral power [uV]')
    if channels is not None:
        plt.legend(channels)
    plt.plot(f, mpow.mean(axis=0))


def rox_burst_duration(bdata, bthresh, sfreq, min_length):
    i = np.flatnonzero(np.diff(np.array(1 * (bdata >= bthresh))) == 1) + 1
    s = np.flatnonzero(np.diff(np.array(1 * (bdata >= bthresh))) == -1) + 1
    if not any(i):
        bdur, n, bamp, peaktime, btime, ibidur, ibiamp = np.nan, 0, np.nan, np.nan, np.nan, np.nan, np.nan
        return bdur, bamp, n, btime, ibidur
    if s[0] < i[0]:
        i = np.append(0, i)
    i = i[0:len(s)]

    istart, istop, bdur, ibidur = i, s, (s - i) / sfreq * 1000, (i[1:] - s[:-1] - 1) / sfreq * 1000
    bamp, ibiamp, btime, peaktime = np.array([]), np.array([]), np.array([]), np.array([])
    for n, dur in enumerate(bdur):
        bamp = np.append(bamp, bdata[istart[n]:istop[n] + 1].sum())
        if n > 0:
            ibiamp = np.append(ibiamp, bdata[istop[n - 1]:istart[n] + 1].mean())
        peaktime = np.append(peaktime, istart[n] - 1 +
                             np.where(
                                 bdata[istart[n]:istop[n] + 1] == np.amax(bdata[istart[n]:istop[n] + 1], axis=None)))
        btime = np.append(btime, istart[n])
    ib = np.flatnonzero(stats.zscore(bamp) < 5)
    if min_length is not None:
        ib = ib[np.isin(ib, np.flatnonzero(bdur > min_length))]
    bamp, btime, peaktime, bdur, n = bamp[ib], btime[ib], peaktime[ib], bdur[ib], len(ib)

    if not any(ib):
        bdur, n, bamp, peaktime, btime, ibidur, ibiamp = np.nan, 0, np.nan, np.nan, np.nan, np.nan, np.nan
    return {'bdur': bdur, 'n': n, 'bamp': bamp, 'peaktime': peaktime, 'btime': btime, 'ibidur': ibidur,
            'ibiamp': ibiamp}


# MNE based functions
def mne_write_edf(mne_raw, fname, picks=None, tmin=0, tmax=None, overwrite=False):
    """
    Saves the raw content of an MNE.io.Raw and its subclasses to
    a file using the EDF+/BDF filetype
    pyEDFlib is used to save the raw contents of the RawArray to disk
    Parameters
    ----------
    mne_raw : mne.io.Raw
        An object with super class mne.io.Raw that contains the data
        to save
    fname : string
        File name of the new dataset. This has to be a new filename
        unless data have been preloaded. Filenames should end with .edf
    picks : array-like of int | None
        Indices of channels to include. If None all channels are kept.
    tmin : float | None
        Time in seconds of first sample to save. If None first sample
        is used.
    tmax : float | None
        Time in seconds of last sample to save. If None last sample
        is used.
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    """
    if not issubclass(type(mne_raw), mne.io.BaseRaw):
        raise TypeError('Must be mne.io.Raw type')
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')
    # static settings
    if os.path.splitext(fname)[-1] == '.edf':
        file_type = pyedflib.FILETYPE_EDFPLUS
        dmin, dmax = -32768, 32767
    else:
        file_type = pyedflib.FILETYPE_BDFPLUS
        dmin, dmax = -8388608, 8388607
    sfreq = mne_raw.info['sfreq']
    date = _stamp_to_dt(mne_raw.info['meas_date'])
    date = date.strftime('%d %b %Y %H:%M:%S')
    first_sample = int(sfreq * tmin)
    last_sample = int(sfreq * tmax) if tmax is not None else None

    # convert data
    channels = mne_raw.get_data(picks,
                                start=first_sample,
                                stop=last_sample)

    # convert to microvolts to scale up precision
    channels *= 1e6

    # set conversion parameters
    pmin, pmax = [channels.min(), channels.max()]
    n_channels = len(channels)

    # create channel from this
    try:
        f = pyedflib.EdfWriter(fname,
                               n_channels=n_channels,
                               file_type=file_type)

        channel_info = []
        data_list = []

        for i in range(n_channels):
            ch_dict = {'label': mne_raw.ch_names[i],
                       'dimension': 'uV',
                       'sample_rate': sfreq,
                       'physical_min': pmin,
                       'physical_max': pmax,
                       'digital_min': dmin,
                       'digital_max': dmax,
                       'transducer': '',
                       'prefilter': ''}

            channel_info.append(ch_dict)
            data_list.append(channels[i])

        f.setTechnician('mne-gist-save-edf-skjerns')
        f.setSignalHeaders(channel_info)
        f.setStartdatetime(date)
        f.writeSamples(data_list)
    except Exception as e:
        print(e)
        return False
    finally:
        f.close()
    return True


def mne_write_bids(filename, subject, bids_folder='/bids', task='rest'):
    raw = mne.io.read_raw_brainvision(str(pathlib.Path(filename)))
    print(raw.annotations)
    raw.annotations.duration[raw.annotations.description == 'DC Correction/'] = 6
    events, event_id = mne.events_from_annotations(raw)  # this needs to be checked
    write_raw_bids(raw, make_bids_basename(subject=subject, task=task), bids_folder, overwrite=True)


def mne_read_bids(filename, bids_folder):
    raw = read_raw_bids(filename, bids_folder)


def mne_events_from_times(times, sfreq, event_id=1):
    if isinstance(event_id, type(1)):
        event_id = [event_id] * len(times)
    events = [int(times[0] * sfreq), 0, event_id[0]]
    for n, t in enumerate(times[1:], 1):
        events = np.vstack((events, [int(t * sfreq), 0, event_id[n]]))
    return events


def mne_import_raw(data, ch_names, sfreq, ch_types='eeg', meas_date=None):
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq
    )
    raw = mne.io.RawArray(data, info)
    if meas_date is not None:
        raw.info['meas_date'] = meas_date
    return raw


def mne_annotations_write_tsv(filename, annot):
    keys, values = [], []
    keys = ['N'] + list(annot[0].keys())
    with open(filename, "w", newline='') as outfile:
        csvwriter = csv.writer(outfile, delimiter='\t')
        csvwriter.writerow(keys)
        for n, i in enumerate(annot, start=1):
            csvwriter.writerow([str(n).zfill(3)] + list(i.values()))


def mne_annotations_read_tsv(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        onset, duration, description, orig_time = [], [], [], []
        for n, lines in enumerate(csv_reader):
            if lines and n:
                onset.append(lines[1])
                duration.append(lines[2])
                description.append(lines[3])
                orig_time.append(lines[4])
    return mne.Annotations(onset, duration, description)


def mne_annotations_get_bad(annot):
    i = np.flatnonzero(np.char.startswith(annot.description, 'BAD'))
    return annot[i]


def mne_annotations_replace_dc(raw):
    raw.annotations.duration[raw.annotations.description == 'DC Correction/'] = 6
    raw.annotations.onset[raw.annotations.description == 'DC Correction/'] = raw.annotations.onset[
                                                                                 raw.annotations.description == 'DC Correction/'] - 2
    raw.annotations.description[raw.annotations.description == 'DC Correction/'] = 'BAD'
    return raw


def mne_read_artifacts(raw, filename):
    artifacts = mne_annotations_read_tsv(filename)
    artifacts.orig_time = raw.annotations.orig_time
    raw.set_annotations(artifacts + raw.annotations)
    return raw


def mne_epoch(raw, tmax=3., tmin=None, events=None, event_id='rest', reject_by_annotation=True):
    if events is None:
        events = mne.make_fixed_length_events(raw, duration=tmax)
        event_id = {'rest': 1}
        tmin = 0
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None,
                        reject_by_annotation=reject_by_annotation)
    return epochs


def mne_cont_epoch(raw, reject_by_annotation=False, event_id='cont'):
    if reject_by_annotation:
        raw = mne_crop_artifacts(raw)
    return mne_epoch(raw, tmax=raw._last_time, tmin=None, event_id=event_id, reject_by_annotation=False)


def mne_mark_flat(raw, picks='eeg'):
    return mne.preprocessing.mark_flat(raw, picks=picks, verbose=True)


def mne_filter(raw, freqs=None):
    raw.load_data()
    if freqs is None:
        raw.notch_filter(np.arange(50, int(raw.info['sfreq']) + 1, 50), method='spectrum_fit', n_jobs=tb.n_jobs())
    elif isinstance(freqs, type([])) and len(freqs) == 2:
        raw.filter(l_freq=freqs[0], h_freq=freqs[1], n_jobs=tb.n_jobs())
    return raw


def mne_resample(raw, sfreq):
    raw.load_data()
    raw.resample(sfreq=sfreq, npad='auto', n_jobs=tb.n_jobs(), events=None)
    return raw


def mne_get_rejection_threshold(raw):
    epochs = mne_epoch(raw)
    epochs.drop_bad()
    return get_rejection_threshold(epochs)


def mne_eeg_remove_eyeblinks(raw, channels=None, overwrite=False):
    if channels is None:
        channels = ['Fp1', 'Fp2']
    icafile = tb.fileparts(raw.filenames[0], '-blink_ica.fif')
    if not pathlib.Path.is_file(pathlib.Path(icafile)) or overwrite:
        filt_raw = raw.copy()
        filt_raw.load_data().filter(l_freq=1., h_freq=None, n_jobs=tb.n_jobs())
        ica = mne.preprocessing.ICA(n_components=15, random_state=97)
        ica.fit(filt_raw)
        ica.exclude = []
        eog_indices = []
        for ch in channels:
            eog_index, eog_scores = ica.find_bads_eog(raw, ch_name=ch, reject_by_annotation=True)
            eog_indices.extend(eog_index)
        eog_indices = list(set(eog_indices))
        ica.exclude = eog_indices
        if eog_indices:
            ica.plot_properties(raw, picks=eog_indices)
            ica.plot_sources(raw)
    else:
        ica = mne.preprocessing.read_ica(icafile)
    clean_raw = raw.copy().load_data()
    ica.apply(clean_raw)
    ica.save(tb.fileparts(raw.filenames[0], '-blink_ica.fif'))
    return clean_raw


def mne_ransac_bad_channels(raw, overwrite=False):
    bids_chan_file = tb.fileparts(raw.filenames[0], '_channels.tsv', -4)
    ransacfile = tb.fileparts(raw.filenames[0], '_channels_ransac.tsv')
    if not pathlib.Path.is_file(pathlib.Path(ransacfile)) or overwrite:
        epochs = mne_epoch(raw).drop_bad()
        epochs.load_data()
        ransac = Ransac(random_state=999)
        ransac.fit(epochs)
        raw.info['bads'] = ransac.bad_chs_
        chans = pd.read_csv(bids_chan_file, delimiter='\t')
        chans.loc[chans.name.isin(ransac.bad_chs_), 'status'] = 'bad'
        pd.DataFrame.to_csv(chans, ransacfile, sep='\t')
    else:
        chans = pd.read_csv(ransacfile, delimiter='\t')
        raw.info['bads'] = list(chans.loc[chans['status'] == 'bad', 'name'])
    return raw


def mne_eeg_inspect_data_quality(raw, overwrite=False):
    events, event_id = mne.events_from_annotations(raw)
    psd_fig = raw.plot_psd(picks='eeg', fmin=2, fmax=40, n_fft=int(3 * raw.info['sfreq']),
                           reject_by_annotation=True)
    psd_fig.savefig(tb.fileparts(raw.filenames[0], '_raw_psd.png'))
    decim = int(raw.info['sfreq'] / 100)
    rawfig = raw.plot(duration=60, block=True, events=events, event_id=event_id, decim=decim, n_channels=32)
    rawfig.savefig(tb.fileparts(raw.filenames[0], '_raw_marked.png'))

    ica = mne_plot_ica(raw)
    ica.plot_sources(raw, block=True)

    ica.save(tb.fileparts(raw.filenames[0], '_visual_ica.fif'))
    print('Please check the data quality and oscillations.')
    assessment = ['usable', 'rating', 'central_theta', 'occipital_alpha', 'central_mu', 'frontal_lowbeta',
                  'frontal_highbeta']
    questions = ['Is this file usable for your purpose?',
                 'Please rate the overall data quality [0-10]',
                 'What is the central theta [4 - 8] peak frequency? [0 if none]',
                 'What is the occipital alpha [8 - 12] peak frequency? [0 if none]',
                 'What is the central mu [8 - 12] peak frequency? [0 if none]',
                 'What is the frontal low beta [13 - 19] peak frequency? [0 if none]',
                 'What is the frontal high beta [20 - 35] peak frequency? [0 if none]']
    vals = []
    for n, a in enumerate(assessment):
        answer = input(questions[n])
        vals.append(answer)
    dataquality = dict(zip(assessment, vals))
    good_samples = np.ones(raw.n_times)
    bad = mne_annotations_get_bad(raw.annotations)
    for n, a in enumerate(bad):
        good_samples[int(raw.time_as_index(a['onset'])):int(raw.time_as_index(a['onset'] + a['duration']))] = 0
    dataquality.update({'good_seconds': np.round(np.sum(good_samples) / raw.info['sfreq'], 2)})
    dataquality.update({'pct_good_samples': np.round(100 * np.sum(good_samples) / raw.n_times, 2)})
    dataquality.update({'good_channels': raw.info['nchan'] - len(raw.info['bads'])})
    dataquality.update({'bad_channels': raw.info['bads']})
    tb.json_write(tb.fileparts(raw.filenames[0], '_dataquality.json'), dataquality)
    plt.close('all')
    return raw


def mne_mark_visual_artifacts(raw, overwrite=False, mark_flat=True, remove_eye_blinks=True, overwrite_ransac=False):
    filename = tb.fileparts(raw.filenames[0], append='_artifacts.tsv')
    channel_file = tb.fileparts(raw.filenames[0], '_channels_marked.tsv', -4)
    event_file = tb.fileparts(raw.filenames[0], '_events_marked.tsv', -4)
    raw.load_data()
    raw = mne_annotations_replace_dc(raw)
    if mark_flat:
        raw = mne_mark_flat(raw, 'eeg')
    if remove_eye_blinks:
        raw = mne_eeg_remove_eyeblinks(raw)
    raw = mne_ransac_bad_channels(raw, overwrite=overwrite_ransac)
    if not pathlib.Path.is_file(pathlib.Path(filename)) or overwrite:
        raw = mne_read_artifacts(raw, filename)
        raw = mne_eeg_inspect_data_quality(raw, overwrite=True)
        i = np.flatnonzero(np.char.startswith(raw.annotations.description, 'BAD'))
        mne_annotations_write_tsv(filename, raw.annotations[i])
        mne_annotations_write_tsv(event_file, raw.annotations)
        chans = pd.read_csv(tb.fileparts(raw.filenames[0], '_channels.tsv', -4), delimiter='\t')
        chans.loc[chans.name.isin(raw.info['bads']), 'status'] = 'bad'
        pd.DataFrame.to_csv(chans, channel_file, sep='\t', index=False)
    else:
        chans = pd.read_csv(channel_file, delimiter='\t')
        raw.info['bads'] = list(chans.loc[chans['status'] == 'bad', 'name'])
        artifacts = mne_annotations_read_tsv(filename)
        artifacts.orig_time = raw.annotations.orig_time
        raw.set_annotations(artifacts + raw.annotations)
    return raw


def mne_bad_channels_from_tsv(bids_channel_file):
    chans = pd.read_csv(bids_channel_file, delimiter='\t')
    return list(chans.loc[chans['status'] == 'bad', 'name'])


def mne_apply_artifact_marks(raw):
    channel_file = tb.fileparts(raw.filenames[0], '_channels_marked.tsv', -4)
    event_file = tb.fileparts(raw.filenames[0], '_events_marked.tsv', -4)
    annotations = mne_annotations_read_tsv(event_file)
    annotations.orig_time = raw.annotations.orig_time
    raw.set_annotations(raw.annotations + annotations)
    raw.info['bads'] = mne_bad_channels_from_tsv(channel_file)
    return raw


def mne_crop_artifacts(raw):
    raw = mne_annotations_replace_dc(raw)
    bad = mne_annotations_get_bad(raw.annotations)
    start = 0
    stop = bad.onset[0]
    cropped_raw = raw.copy().crop(start, stop)
    for n, a in enumerate(bad, 0):
        if n + 1 < len(bad):
            stop = bad.onset[n + 1]
        else:
            stop = raw._last_time

        new_start = bad.onset[n] + bad.duration[n]
        if new_start > start:
            start = new_start

        if stop > raw._last_time:
            stop = raw._last_time
        if 0 < start < stop < raw._last_time:
            cropped_raw = mne.concatenate_raws([cropped_raw, raw.copy().crop(start, stop)])
            print((start, stop))
    return cropped_raw


def mne_set_1020_montage(raw):
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    return raw


def mne_lcmv_template(raw, apply_to=None, fmin=3, fmax=45):
    # set paths and filenames
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    subjects_dir = mne.get_config('SUBJECTS_DIR')
    fs_dir = str(pathlib.Path(subjects_dir, 'fsaverage'))
    src = str(pathlib.Path(fs_dir, 'bem', 'fsaverage-ico-5-src.fif'))
    bem = str(pathlib.Path(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif'))
    # prepare raw data
    raw.load_data()
    raw = mne_filter(raw, [fmin, fmax])
    raw = mne_set_1020_montage(raw)
    raw.interpolate_bads()
    raw.set_eeg_reference('average', projection=True)
    # prepare mne beamforming solutions
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=tb.n_jobs())
    # cov = mne.compute_raw_covariance(raw)  # compute before band-pass of interest
    epochs = mne_epoch(raw, 3.)
    epochs.load_data()
    cov = mne.compute_covariance(epochs)
    filters = mne.beamformer.make_lcmv(raw.info, fwd, cov)
    # , 0.05, noise_cov=None, pick_ori='max-power', weight_norm='nai')
    if apply_to is None:
        raw_lcmv = mne.beamformer.apply_lcmv_raw(raw, filters)
    else:
        apply_to = mne_set_1020_montage(apply_to)
        apply_to.interpolate_bads()
        apply_to.set_eeg_reference('average', projection=True)
        raw_lcmv = mne.beamformer.apply_lcmv_raw(apply_to, filters)
    return raw_lcmv
    # src = mne.setup_source_space(subject, spacing='ico4',subjects_dir=subjects_dir, add_dist=False)
    # mne.write_source_spaces(pth(fs_dir,'bem','fsaverage-ico-4-src.fif'), src,overwrite=True)
    # model = mne.make_bem_model('fsaverage')
    # mne.write_bem_surfaces(pth(fs_dir,'bem','fsaverage-5120-5120-5120-bem.fif'), model,overwrite=True)
    # bem_sol = mne.make_bem_solution(model)
    # bem=mne.write_bem_solution('fsaverage-5120-5120-5120-bem-sol.fif', bem_sol,overwrite=True)
    # mne.viz.plot_alignment(reconst_raw.info, src=src, eeg=['original', 'projected'], trans=trans,show_axes=True, mri_fiducials=True, dig='fiducials')


def mne_lcmv_to_raw(lcmv):
    ch_types, ch_names = [], []
    for a in np.arange(0, lcmv.shape[0]):
        ch_types.append('seeg')
        ch_names.append('lcmv' + str(a))
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=lcmv.sfreq)
    raw = mne.io.RawArray(lcmv.data, info)
    raw.info['surface'] = lcmv.mean()
    return raw


def mne_tf_wavelet(inst, freqs=np.arange(3, 100), n_freq=20, n_cycles=7., zero_mean=True):
    wav = mne.time_frequency.tfr_morlet(inst, freqs, n_cycles=n_cycles, zero_mean=zero_mean, use_fft=False,
                                        return_itc=False,
                                        decim=int(inst.info['sfreq'] / n_freq), n_jobs=1, picks=None,
                                        average=False, output='power', verbose='INFO')
    return wav


def mne_icn_rest(inst):
    coh = pd.DataFrame
    pow, f = mne.time_frequency.psd_multitaper(inst, fmin=1, fmax=99)
    mpow = pow.mean(axis=0)
    rpow = mpow.copy()
    spow = mpow.copy()
    fnorm = np.append(np.arange(f.searchsorted(5), f.searchsorted(45) + 1),
                      np.arange(f.searchsorted(55), f.searchsorted(95) + 1))
    for n in np.arange(0, mpow.shape[0]):
        rpow[n, :] = (mpow[n, :] / mpow[n, fnorm].sum()) * 100
        spow[n, :] = (mpow[n, :] / mpow[n, fnorm].std())
    con = mne.connectivity.spectral_connectivity(inst, method='coh', fmin=1, fmax=99)
    coh = con[0]
    con = mne.connectivity.spectral_connectivity(inst, method='imcoh', fmin=1, fmax=99)
    icoh = con[0]
    channels = inst.ch_names

    # todo: add shuffling and return


def mne_burst_analysis(raw, freqranges=None, threshold=50, common_threshold=True, min_length=200, method='wavelet',
                       smoothing=200, common_zscore=True):
    if freqranges is None:
        freqranges = {'beta': [13, 30]}
    csvfile = tb.fileparts(raw.filenames[0], '_bursts.tsv')
    bdf = pd.DataFrame([])
    for fband, frange in freqranges.items():
        print(fband)
        if method == 'hilbert':
            df = raw.copy().filter(l_freq=frange[0], h_freq=frange[1], n_jobs=tb.n_jobs()).apply_hilbert(
                envelope=True).to_data_frame()
            for ch in raw.ch_names:
                df.loc[:, ch] = df[ch].rolling(axis=0, window=int(smoothing / 1000 * raw.info['sfreq'])).mean().astype(
                    'float').interpolate(
                    method='linear', limit_direction='both')
            sfreq = raw.info['sfreq']
            info = raw.info
        elif method == 'wavelet':
            wav = mne_tf_wavelet(mne_cont_epoch(raw), freqs=np.arange(frange[0], frange[1] + 1),
                                 n_freq=smoothing / 1000 * raw.info['sfreq'], zero_mean=False, n_cycles=10)
            df = pd.DataFrame(np.squeeze(wav.data.mean(axis=2)).transpose(), columns=raw.ch_names)
            sfreq = wav.info['sfreq']
            info = wav.info
        if common_zscore:
            df = (df - df.mean()) / df.std()

        bursts = OrderedDict()
        if common_threshold:
            bthresh = np.percentile(df, threshold)

        for ch in raw.ch_names:
            bdata = df.loc[:, ch]
            if not common_threshold:
                bthresh = np.percentile(bdata, threshold)
            bursts.update({ch: rox_burst_duration(bdata, bthresh, sfreq, min_length)})
            bdf.loc[ch, fband + '_threshold'] = bthresh
            if bursts[ch]['n']:
                if bursts[ch]['n'] > 10:
                    mdl = stats.fitlm_kfold(bursts[ch]['bdur'], bursts[ch]['bamp'], 5)
                    bdf.loc[ch, fband + '_slope'] = np.mean(mdl[1])
                else:
                    bdf.loc[ch, fband + '_slope'] = 0
                bdf.loc[ch, fband + '_mdur'] = bursts[ch]['bdur'].mean()
                bdf.loc[ch, fband + '_n'] = bursts[ch]['n'] / raw._last_time
                bdf.loc[ch, fband + '_mamp'] = bursts[ch]['bamp'].mean()
                bdf.loc[ch, fband + '_mpow'] = bdata.mean()
            else:
                for s in ['_slope', '_mdur', '_n', '_mamp', '_mpow']:
                    bdf.loc[ch, fband + s] = 0

    bdf.to_csv(csvfile, sep='\t')
    burst_settings = {'threshold[%]': threshold, 'common_threshold': common_threshold,
                      'common_zscore': common_zscore, 'sfreq[Hz]': sfreq, 'method': method, 'smoothing[ms]': smoothing,
                      'min_length[ms]': min_length, 'freqranges[Hz]': freqranges}
    tb.json_write(tb.fileparts(raw.filenames[0], '_bursts.json'), burst_settings)
    bdf._metadata = burst_settings
    return ['df', 'setttings', 'info', 'chanwise'], bdf, burst_settings, info, bursts
    # bdf.loc[ch, fband + '_hist'] = str(np.histogram(bursts[ch]['bdur'], np.arange(min_length,
    # 1000+min_length, 100))[0])


# MNE plotting wrappers
def mne_plot_raw(raw, block=False, outfile=None, show=True):
    events, event_id = mne.events_from_annotations(raw)
    decim = int(raw.info['sfreq'] / 100)
    raw_fig = raw.plot(events=events, event_id=event_id, butterfly=True, decim=decim, group_by='position',
                       lowpass=99, highpass=1, filtorder=0, show_options=True, scalings={'eeg': 25e-6, 'misc': 1},
                       block=block, title=raw.filenames[0], remove_dc=True, show=show)
    if outfile:
        raw_fig.savefig(outfile)
    return raw_fig


def mne_plot_psd_topo(raw, block=False, outfile=None, show=True):
    raw_fig = raw.plot_psd_topo(fmin=0.1, fmax=40, n_fft=int(3 * raw.info['sfreq']), block=block, show=show)
    if outfile:
        raw_fig.savefig(outfile, facecolor='black')
    return raw_fig


def mne_plot_burst_topomaps(df, info, burst_settings):
    if burst_settings is None:
        burst_settings = df._metadata
    for fb in burst_settings['freqranges[Hz]'].keys():
        plt.figure(figsize=(14, 5))
        i = tb.ci(fb, list(df.keys()))
        for n, a in enumerate(i, 1):
            plt.subplot(1, len(i), n)
            mne.viz.plot_topomap(stats.zscore(df.iloc[:, a]), info, cmap='viridis')
            plt.title(str.replace(df.keys()[a], '_', ' '))


def mne_plot_lcmv(lcmv, surface='orig', hemi='both', smoothing_steps=10, transparent=True, alpha=1,
                  time_viewer=True, views='dor', verbose=True, cortex=None, spacing='ico4'):
    if cortex is None:
        cortex = [0.8, 0.8, 0.8]
    fig = lcmv.plot(surface=surface, hemi=hemi, smoothing_steps=smoothing_steps, transparent=transparent, alpha=alpha,
                    time_viewer=time_viewer, views=views, verbose=verbose, cortex=cortex, spacing=spacing)
    return fig


def mne_plot_ica(raw, n_components=15):
    ica = mne.preprocessing.ICA(n_components=n_components)
    ica.fit(raw)
    ica_raw = ica.get_sources(mne_crop_artifacts(raw))
    mapping = dict()
    for a in np.arange(0, ica_raw.info['nchan']):
        mapping.update({ica_raw.ch_names[a]: 'eeg'})
    ica_raw.set_channel_types(mapping)
    pow, f = mne.time_frequency.psd_welch(ica_raw, n_fft=int(3 * ica_raw.info['sfreq']), fmin=2, fmax=40)
    mpow = pow.mean(axis=0)
    for a in np.arange(0, pow.shape[0]):
        fig = ica.plot_components(a)
        fig.set_size_inches(8, 5)
        fig.axes[0].change_geometry(1, 2, 1)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(f, pow[a, :])
        ax.plot(f, mpow, color='red')
        ax.plot(f, pow[a, :] - mpow, color='green')
        ax.set_position([0.55, 0.1, 0.4, 0.7])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power')
        plt.legend([ica_raw.ch_names[a], 'avg', 'diff'])
    return ica
