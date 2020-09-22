import os
import pathlib
import shutil
import fileinput
import dateutil
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne_bids import write_raw_bids, make_bids_basename, make_bids_folders

import icn_ephys as ephys
import icn_tb as tb


def read_file(filename):
    return pd.read_json(filename, typ='Series',convert_dates=False)


def patient_name(filename):
    data = read_file(filename)
    return data.PatientInformation['Final']['PatientLastName'], data.PatientInformation['Final']['PatientFirstName']


def anonymize(filename):
    [fdir,fname,ext] = tb.fileparts(filename)
    copyname = pathlib.Path(fdir,'Sensitive_'+fname[7:]+ext)
    shutil.copyfile(filename,copyname)
    data = read_file(filename)
    ainfo = []
    ainfo.append(data.PatientInformation['Final']['PatientLastName'])
    ainfo.append(data.PatientInformation['Final']['PatientFirstName'])
    ainfo.append(data.PatientInformation['Final']['PatientId'])
    ainfo.append(data.PatientInformation['Final']['PatientDateOfBirth'])
    ainfo.append(data.PatientInformation['Final']['PatientGender'])
    ainfo.append(data.DeviceInformation['Final']['NeurostimulatorSerialNumber'])

    for a in ainfo:
        tb.replace_txt_in_file(filename, a)
    os.remove(str(filename)+'.bak')
    return filename


def reformat_LFPMontage_channelname(LFPMontage, target='LFP'):
    n = LFPMontage['SensingElectrodes'].split('.')[-1].replace('AND', '') \
        .replace('_', '') \
        .replace('ZERO', '0') \
        .replace('ONE', '1') \
        .replace('TWO', '2') \
        .replace('THREE', '3') \
        .replace('FOUR', '4')
    side = LFPMontage['Hemisphere'].split('.')[-1][0]
    return target + side + n


def reformat_LfpMontageTimeDomain_channelname(LfpMontageTimeDomain, target='LFP'):
    n = LfpMontageTimeDomain['Channel'].replace('_AND','') \
        .replace('ZERO', '0') \
        .replace('ONE', '1') \
        .replace('TWO', '2') \
        .replace('THREE', '3') \
        .replace('FOUR', '4').split('_')
    n = n[0]+n[1]
    side = LfpMontageTimeDomain['Channel'].split('_')[-2][0]
    return target + side + n


def reformat_BrainSense_channelname(BrainSense, target='LFP'):
    n = len(BrainSense)
    chans = list()
    for a in BrainSense:
        ch = a['Channel'] \
            .replace('ZERO', '0') \
            .replace('ONE', '1') \
            .replace('TWO', '2') \
            .replace('THREE', '3') \
            .replace('FOUR', '4').split('_')
        chans.append(target + ch[2][0] + ch[0] + ch[1])
    return chans


def reformat_BrainSenseTimeDomain_channelname(BrainSenseTimeDomain, target='LFP'):
    ch =BrainSenseTimeDomain['Channel'] \
        .replace('ZERO', '0') \
        .replace('ONE', '1') \
        .replace('TWO', '2') \
        .replace('THREE', '3') \
        .replace('FOUR', '4').split('_')
    return target + ch[2][0] + ch[0] + ch[1]


def reformat_BrainSenseLfp_channelname(bs, target='LFP'):
    ch = bs['Channel'].replace('ZERO', '0') \
            .replace('ONE', '1') \
            .replace('TWO', '2') \
            .replace('THREE', '3') \
            .replace('FOUR', '4') \
            .replace('_','').split(',')
    return target + ch[0][2] + ch[0][0:2], target + ch[1][2] + ch[1][0:2]


def reformat_DateTime(DateTime):
    return DateTime[:-1].replace('-', '').replace(':', '')


def reformat_LfpFrequencySnapshotEvents_channelname(SenseID, hemisphere, target='LFP'):
    ch = SenseID.split('.')[-1].replace('AND', '') \
        .replace('_', '') \
        .replace('ZERO', '0') \
        .replace('ONE', '1') \
        .replace('TWO', '2') \
        .replace('THREE', '3') \
        .replace('FOUR', '4').split('_')
    return target + hemisphere + ch[0]


def plot_LfpFrequencySnapshotEvents(filename):
    data = read_file(filename)
    fpath = tb.fileparts(filename)
    if 'LfpFrequencySnapshotEvents' in data['DiagnosticData']:
        eventlist = data['DiagnosticData']['LfpFrequencySnapshotEvents']
        n = 0
        for a in eventlist:
            if 'LfpFrequencySnapshotEvents' in a.keys():
                if n == 0:
                    spectra = pd.DataFrame(
                        index=a['LfpFrequencySnapshotEvents']['HemisphereLocationDef.Right']['Frequency'])
                n += 1
                outname = reformat_DateTime(a['DateTime']) + '_' + a['EventName'].replace(' ', '_') + '_ID' + str(
                    a['EventID'])
                lfpr = a['LfpFrequencySnapshotEvents']['HemisphereLocationDef.Right']
                lfpl = a['LfpFrequencySnapshotEvents']['HemisphereLocationDef.Left']
                ch_r = reformat_LfpFrequencySnapshotEvents_channelname(lfpr['SenseID'], 'R')
                ch_l = reformat_LfpFrequencySnapshotEvents_channelname(lfpl['SenseID'], 'L')
                pfig = plt.figure(figsize=(5.17, 4.05))
                plt.plot(lfpr['Frequency'], lfpr['FFTBinData'])
                plt.plot(lfpl['Frequency'], lfpl['FFTBinData'])
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Spectral power [uV]')
                plt.legend([ch_r, ch_l])
                plt.title(outname)
                spectra.loc[:, outname + '_' + ch_r] = lfpr['FFTBinData']
                spectra.loc[:, outname + '_' + ch_l] = lfpr['FFTBinData']
                if not os.path.isdir(pathlib.Path(fpath[0], 'figures')):
                    os.makedirs(pathlib.Path(fpath[0], 'figures'))
                pfig.savefig(str(pathlib.Path(fpath[0], 'figures', outname + '_snapshot_' + str(n) + '.png')), dpi=300)
        spectra.to_csv(pathlib.Path(fpath[0], 'figures', outname + '_snapshot_psd.tsv'), sep='\t')


def import_LfpMontageTimeDomain(filename):
    data = read_file(filename)

    if 'LfpMontageTimeDomain' in data.keys():
        sfreq = data['LfpMontageTimeDomain'][0]['SampleRateInHz']
        chans, hemisphere = [], []
        for fmtd in data['LfpMontageTimeDomain']:
            cchan = reformat_LfpMontageTimeDomain_channelname(fmtd)
            chans.append(cchan)
            hemisphere.append(cchan[3])
        uchans = np.unique(chans)

        raw_data = pd.DataFrame()
        raw_data.columns.name = 'Channels'

        for ch in uchans:
            i = tb.ci(ch, chans)
            cdata = data['LfpMontageTimeDomain'][i[0]]['TimeDomainData']
            if len(i) > 1:
                for ni in np.arange(1, len(i)):
                    ndata = data['LfpMontageTimeDomain'][i[ni]]['TimeDomainData']
                    cdata = np.concatenate((cdata, ndata))
            if isinstance(cdata, type([])):
                raw_data.loc[:, ch] = cdata
            else:
                raw_data.loc[:, ch] = cdata.transpose()

        meas_date = dateutil.parser.parse(data['LfpMontageTimeDomain'][0]['FirstPacketDateTime'].replace('T', ' ').replace('0Z', ''))
        raw = ephys.mne_import_raw(raw_data.values.transpose() / 100000, list(raw_data.keys()), sfreq=sfreq,
                                   ch_types='seeg', meas_date=(meas_date.timestamp(), 0))
        return raw
    else:
        return None


def import_BrainSenseTimeDomain(filename):
    data = read_file(filename)
    if 'BrainSenseTimeDomain' in data.keys():
        sfreq = data['BrainSenseTimeDomain'][0]['SampleRateInHz']
        chans, hemisphere = [], []
        for bstd in data['BrainSenseTimeDomain']:
            cchan = reformat_BrainSenseTimeDomain_channelname(bstd)
            chans.append(cchan)
            hemisphere.append(cchan[3])
        uchans = np.unique(chans)

        raw_data = pd.DataFrame()
        raw_data.columns.name = 'Channels'
        ticks = []
        for n, ch in enumerate(uchans):
            i = tb.ci(ch, chans)
            cdata = data['BrainSenseTimeDomain'][i[0]]['TimeDomainData']
            if n==0:
                ticks = np.double(data['BrainSenseTimeDomain'][i[0]]['TicksInMses'][:-1].split(','))
            if len(i) > 1:
                for ni in np.arange(1, len(i)):
                    ndata = data['LfpMontageTimeDomain'][i[ni]]['TimeDomainData']
                    cdata = np.concatenate((cdata, ndata))
            if isinstance(cdata,type([])):
                raw_data.loc[:, ch] = cdata
            else:
                raw_data.loc[:, ch] = cdata.transpose()

        description, duration, onset = [],[],[]
        ne = 0
        tadd = 0
        times = []
        for nbs, bs in enumerate(data['BrainSenseLfp']):
            rp = bs['TherapySnapshot']['Right']
            lp = bs['TherapySnapshot']['Left']
            rsp = str(rp['PulseWidthInMicroSecond']) + '_us_' + str(rp['RateInHertz']) + '_Hz'
            lsp = str(lp['PulseWidthInMicroSecond']) + '_us_' + str(lp['RateInHertz']) + '_Hz'
            for n, events in enumerate(bs['LfpData']):
                if n==0 and nbs > 0:
                    tadd = onset[-1] + duration[-1]
                if n == 0:
                    first_onset = bs['LfpData'][0]['TicksInMs']
                    stim_duration = (bs['LfpData'][1]['TicksInMs']-first_onset)/1000
                ne+=1
                rss = str(events['Right']['LFP']) + '_mV_' + str(events['Right']['mA']) + '_mA'
                lss = str(events['Left']['LFP']) + '_mV_' + str(events['Left']['mA']) + '_mA'
                desc = 'R_' + rss + '_' + rsp + '_L_' + lss + '_' + lsp
                description.append(desc)
                duration.append(stim_duration)
                times.append(events['TicksInMs'])
                onset.append(tadd+(events['TicksInMs']-first_onset)/1000)

        meas_date = dateutil.parser.parse(data['BrainSenseTimeDomain'][0]['FirstPacketDateTime'].replace('T', ' ').replace('0Z', ''))
        raw = ephys.mne_import_raw(raw_data.values.transpose() / 100000, list(raw_data.keys()), sfreq=sfreq,
                                   ch_types='seeg', meas_date=(meas_date.timestamp(), 0))
        # raw.set_annotations(mne.Annotations(onset,duration,description,None)) bug
        return raw
    else:
        return None



def import_IndefiniteStreaming(filename):
    data = read_file(filename)
    if 'IndefiniteStreaming' in data.keys():
        # concatenate streams
        chans = reformat_BrainSense_channelname(data['IndefiniteStreaming'])
        uchs = np.unique(chans)
        sfreq = data['IndefiniteStreaming'][0]['SampleRateInHz']
        raw_data = pd.DataFrame()
        raw_data.columns.name = 'Channels'
        for ch in uchs:
            i = tb.ci(ch, chans)
            cdata = data['IndefiniteStreaming'][i[0]]['TimeDomainData']
            if len(i) > 1:
                for ni in np.arange(1, len(i)):
                    ndata = data['IndefiniteStreaming'][i[ni]]['TimeDomainData']
                    cdata = np.concatenate((cdata, ndata))
            if isinstance(cdata,type([])):
                raw_data.loc[:,ch] = cdata
            else:
                raw_data.loc[:, ch] = cdata.transpose()
        meas_date = dateutil.parser.parse(
            data['IndefiniteStreaming'][0]['FirstPacketDateTime'].replace('T', ' ').replace('0Z', ''))
        raw = ephys.mne_import_raw(raw_data.values.transpose() / 100000, list(raw_data.keys()), sfreq=sfreq,
                                   ch_types='seeg', meas_date=(meas_date.timestamp(), 0))
        return raw
    else:
        return None


def get_TimeDomainFieldNames(filename=None):
    fnames = ['IndefiniteStreaming', 'BrainSenseTimeDomain', 'LfpMontageTimeDomain']

    if filename is None:
        return fnames
    else:
        data = read_file(filename)
        retlist = list()
        for i in fnames:
            if i in data.keys():
                retlist.append(i)
        return retlist


def import_rawdata(filename, typefield='try'):
    raw = None
    if typefield == 'try' or typefield == 'IndefiniteStreaming':
        raw = import_IndefiniteStreaming(filename)
    if (raw is None and typefield == 'try') or typefield == 'BrainSenseTimeDomain':
        raw = import_BrainSenseTimeDomain(filename)
    if (raw is None and typefield == 'try') or typefield == 'LfpMontageTimeDomain':
        raw = import_LfpMontageTimeDomain(filename)
    return raw


def tf_LfpMontageTimeDomain_wavelet(filename):
    data = read_file(filename)
    raw = import_LfpMontageTimeDomain(filename)
    if raw is not None:
        epochs = ephys.mne_cont_epoch(raw)
        wav = ephys.mne_tf_wavelet(epochs)
        return wav
    else:
        return None, None


def plot_LFPMontage_spectra(filename):
    data = read_file(filename)
    fpath = tb.fileparts(filename)
    if 'LFPMontage' in data.keys():
        chans = list()
        nchans = len(data.LFPMontage)
        hemisphere = list()
        for a in np.arange(0, nchans):
            hemisphere.append(data.LFPMontage[a]['Hemisphere'].split('.')[-1][0])
            chans.append(reformat_LFPMontage_channelname(data.LFPMontage[a]))
        ir, il = tb.ci('R', hemisphere), tb.ci('L', hemisphere)
        right_chans = [chans[i] for i in ir]
        left_chans = [chans[i] for i in il]
        columns = [chans[i] for i in ir + il]
        spectra = pd.DataFrame(columns=columns, index=data.LFPMontage[0]['LFPFrequency'])
        spectra.index.name = 'Frequency'
        spectra.columns.name = 'Channels'
        pfig = plt.figure(figsize=(11.17, 4.05))
        plt.subplot(1, 2, 1)
        for a in ir:
            lfp = data.LFPMontage[a]
            linestyle = ('-', '--')
            artefact = lfp['ArtifactStatus'].split('.')[-1] != 'ARTIFACT_NOT_PRESENT'
            plt.plot(lfp['LFPFrequency'], lfp['LFPMagnitude'], linewidth=0.5, linestyle=linestyle[artefact])

            if not artefact:
                plt.scatter(lfp['PeakFrequencyInHertz'], lfp['PeakMagnitudeInMicroVolt'])
                spectra[chans[a]] = lfp['LFPMagnitude']
        spectra.loc[:, right_chans].mean(axis=1).plot(linewidth=3, color='k')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Spectral power [uV]')
        plt.legend(right_chans)
        plt.title('Right hemisphere')

        plt.subplot(1, 2, 2)
        for a in il:
            lfp = data.LFPMontage[a]
            linestyle = ('-', '--')
            artefact = lfp['ArtifactStatus'] == 'ArtifactStatusDef.ARTIFACT_PRESENT'
            plt.plot(lfp['LFPFrequency'], lfp['LFPMagnitude'], linewidth=0.5, linestyle=linestyle[artefact])
            plt.scatter(lfp['PeakFrequencyInHertz'], lfp['PeakMagnitudeInMicroVolt'])
            spectra[chans[a]] = lfp['LFPMagnitude']
        spectra.loc[:, left_chans].mean(axis=1).plot(linewidth=3, color='k')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Spectral power [uV]')
        plt.legend(left_chans)
        plt.title('Left hemisphere')
        outname = data['SessionDate'][:-1].replace('-', '').replace(':', '')
        plt.suptitle(outname)
        if not os.path.isdir(pathlib.Path(fpath[0], 'figures')):
            os.makedirs(pathlib.Path(fpath[0], 'figures'))
        pfig.savefig(str(pathlib.Path(fpath[0], 'figures', outname + '_LFPMontage_raw.png')), dpi=300)
        spectra.to_csv(pathlib.Path(fpath[0], 'figures', outname + '_LFPMontage_raw.tsv'), sep='\t')
    else:
        print('No LFPMontage in file ' + filename)
        return None, None


def plot_BrainSenseLfp(filename):
    data = read_file(filename)
    fpath = tb.fileparts(filename)
    if 'BrainSenseLfp' in data.keys():
        chans = list()
        nses = len(data.BrainSenseLfp)
        for a in np.arange(0, nses):
            bs = data['BrainSenseLfp'][a]
            chans = reformat_BrainSenseLfp_channelname(bs)
            fs = bs['SampleRateInHz']
            dbs = bs['TherapySnapshot']
            lfp = bs['LfpData']
            df = pd.DataFrame()
            for n, b in enumerate(lfp):
                if n == 0:
                    tmin = b['TicksInMs']
                df.loc[n, 'Time'] = (b['TicksInMs']-tmin)/1000
                df.loc[n, 'AbsTime'] = b['TicksInMs']
                df.loc[n, chans[1]] = b['Right']['LFP']/1000
                df.loc[n, chans[0]] = b['Left']['LFP']/1000
                df.loc[n, 'stimR'] = b['Right']['mA']
                df.loc[n, 'stimL'] = b['Left']['mA']

        ylegend = [chans[1], chans[0], 'stimR', 'stimL']
        df.plot(x='Time', y=ylegend)
        plt.xlabel('Time [s]')
        plt.ylabel('LFP + Stimulation amplitude')
        outname = data['SessionDate'][:-1].replace('-', '').replace(':', '')
        plt.title(outname)

        if not os.path.isdir(pathlib.Path(fpath[0], 'figures')):
            os.makedirs(pathlib.Path(fpath[0], 'figures'))
        plt.gcf().savefig(str(pathlib.Path(fpath[0], 'figures', outname + '_BrainSenseLfp.png')), dpi=300)
        df.to_csv(pathlib.Path(fpath[0], 'figures', outname + '_BrainSenseLfp.tsv'), sep='\t')
    else:
        print('No BrainSenseLfp in file ' + filename)
        return None, None


def plot_wavelet_spectra(filename, typefield='all'):
    data = read_file(filename)
    fpath = tb.fileparts(filename)
    if typefield == 'all':
        flist = get_TimeDomainFieldNames(filename)
    else:
        flist = typefield
        if isinstance(flist, str):
            flist = [flist]
    for f in flist:
        raw = import_rawdata(filename, typefield=f)
        if raw is not None:
            epochs = ephys.mne_cont_epoch(raw)
            wav = ephys.mne_tf_wavelet(epochs)
            mpow = wav.data[0, :, :, :].mean(axis=2)
            rpow, spow = ephys.normalize_spectrum(mpow, wav.freqs)
            pfig = plt.figure(figsize=(11.17, 4.05))
            plt.subplot(1, 2, 1)
            i = tb.ci('LFPR', wav.ch_names)
            for a in i:
                plt.plot(wav.freqs, rpow[a, :].transpose(), linewidth=.5)
            plt.legend([wav.ch_names[ir] for ir in i])
            plt.plot(wav.freqs, rpow[i, :].mean(axis=0), linewidth=2, color='k')
            plt.ylim((0, 10))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Relative spectral power [%]')
            plt.xlim((0, 45))
            plt.title('Right hemisphere')
            plt.subplot(1, 2, 2)
            i = tb.ci('LFPL', wav.ch_names)
            for a in i:
                plt.plot(wav.freqs, rpow[a, :], linewidth=.5)
                plt.title(a)
            plt.legend([wav.ch_names[ir] for ir in i])
            plt.plot(wav.freqs, rpow[i, :].mean(axis=0), linewidth=2, color='k')
            plt.ylim((0, 10))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Relative spectral power [%]')
            plt.xlim((0, 45))
            plt.title('Left hemisphere')
            outname = data['SessionDate'][:-1].replace('-', '').replace(':', '') + '_' + f
            plt.suptitle(outname)
            if not os.path.isdir(pathlib.Path(fpath[0], 'figures')):
                os.makedirs(pathlib.Path(fpath[0], 'figures'))
            pfig.savefig(str(pathlib.Path(fpath[0], 'figures', outname + '_rpow.png')), dpi=300)
            spectra = pd.DataFrame(np.transpose(rpow), index=wav.freqs, columns=wav.ch_names)
            spectra.to_csv(pathlib.Path(fpath[0], 'figures', outname + '_rpow.tsv'), sep='\t')
            spectra = pd.DataFrame(np.transpose(spow), index=wav.freqs, columns=wav.ch_names)
            spectra.to_csv(pathlib.Path(fpath[0], 'figures', outname + '_spow.tsv'), sep='\t')
            spectra = pd.DataFrame(np.transpose(mpow), index=wav.freqs, columns=wav.ch_names)
            spectra.to_csv(pathlib.Path(fpath[0], 'figures', outname + '_mpow.tsv'), sep='\t')


def LfpMontageTimeDomain_to_bids(filename, subject, bids_folder='/bids', task='LfpMontageTimeDomain'):
    data = read_file(filename)
    opath, fname, ext = tb.fileparts(filename)
    if subject.find('-') > 0:
        subject = subject[subject.find('-') + 1:]
    session = data['SessionDate'][:-1].replace('-', '').replace(':', '')
    sourcename = make_bids_basename(subject=subject, session=session)
    basename = make_bids_basename(subject=subject, task=task, session=session)
    bpath = make_bids_folders(subject=subject, session=session, make_dir=False)
    raw = import_LfpMontageTimeDomain(filename)
    if raw is not None:
        rfig = raw.plot(color='k')
        tb.mkdir(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', 'figures'))
        rfig.savefig(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', 'figures', basename + '.png'))

        if os.path.exists('tmp.edf'):
            os.remove('tmp.edf')
        ephys.mne_write_edf(raw, 'tmp.edf')
        raw = mne.io.read_raw_edf('tmp.edf')
        ephys.write_raw_bids(raw, bids_basename=basename, output_path=bids_folder, overwrite=True)
        shutil.copyfile(filename, pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', basename + ext))
        plot_wavelet_spectra(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', basename + ext), typefield=task)
        plot_LFPMontage_spectra(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', basename + ext))
        os.remove('tmp.edf')
        shutil.move(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', basename + ext),
                    pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', sourcename + ext))


def BrainSenseTimeDomain_to_bids(filename, subject, bids_folder='/bids', task='BrainSenseTimeDomain'):
    data = read_file(filename)
    opath, fname, ext = tb.fileparts(filename)
    if subject.find('-') > 0:
        subject = subject[subject.find('-') + 1:]
    session = data['SessionDate'][:-1].replace('-', '').replace(':', '')
    basename = make_bids_basename(subject=subject, task=task, session=session)
    bpath = make_bids_folders(subject=subject, session=session, make_dir=False)
    sourcename = make_bids_basename(subject=subject, session=session)
    raw = import_BrainSenseTimeDomain(filename)
    if raw is not None:
        rfig = raw.plot(color='k')
        rfig.savefig(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', 'figures', basename + '.png'))
        if os.path.exists('tmp.edf'):
            os.remove('tmp.edf')
        ephys.mne_write_edf(raw, 'tmp.edf')
        raw = mne.io.read_raw_edf('tmp.edf')
        write_raw_bids(raw, bids_basename=basename, output_path=bids_folder, overwrite=True)
        if not os.path.isdir(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata')):
            os.makedirs(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata'))
        shutil.copyfile(filename, pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', basename + ext))
        plot_wavelet_spectra(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', basename + ext), typefield=task)
        plot_BrainSenseLfp(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', basename + ext))
        os.remove('tmp.edf')
        shutil.move(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', basename + ext),
                    pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', sourcename + ext))


def IndefiniteStreaming_to_bids(filename, subject, bids_folder='/bids', task='IndefiniteStreaming'):
    data = read_file(filename)
    opath, fname, ext = tb.fileparts(filename)
    if subject.find('-') > 0:
        subject = subject[subject.find('-') + 1:]
    session = data['SessionDate'][:-1].replace('-', '').replace(':', '')
    basename = make_bids_basename(subject=subject, task=task, session=session)
    bpath = make_bids_folders(subject=subject, session=session, make_dir=False)
    sourcename = make_bids_basename(subject=subject, session=session)
    raw = import_IndefiniteStreaming(filename)
    if raw is not None:
        rfig = raw.plot(color='k')
        rfig.savefig(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', 'figures', basename + '.png'))

        if os.path.exists('tmp.edf'):
            os.remove('tmp.edf')

        ephys.mne_write_edf(raw, 'tmp.edf')
        raw = mne.io.read_raw_edf('tmp.edf')
        write_raw_bids(raw, bids_basename=basename, output_path=bids_folder, overwrite=True)
        if not os.path.isdir(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata')):
            os.makedirs(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata'))
        shutil.copyfile(filename, pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', basename + ext))
        plot_wavelet_spectra(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', basename + ext), typefield=task)
        os.remove('tmp.edf')
        shutil.move(pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', basename + ext),
                    pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata', sourcename + ext))


def convert_to_bids(filename, subject, bids_folder):
    subject=subject.replace('sub-','')
    data = read_file(filename)
    session = data['SessionDate'][:-1].replace('-', '').replace(':', '')
    bpath = pathlib.Path(bids_folder, make_bids_folders(subject=subject, session=session, make_dir=False))
    sourcename = make_bids_basename(subject=subject, session=session)
    sourcefolder = pathlib.Path(bids_folder, bpath, 'eeg', 'sourcedata')
    tb.mkdir(sourcefolder)
    shutil.copyfile(filename, pathlib.Path(sourcefolder, sourcename + '.json'))
    plot_LfpFrequencySnapshotEvents(pathlib.Path(sourcefolder, sourcename + '.json'))
    LfpMontageTimeDomain_to_bids(filename, subject, bids_folder)
    BrainSenseTimeDomain_to_bids(filename, subject, bids_folder)
    IndefiniteStreaming_to_bids(filename, subject, bids_folder)
    plt.close('all')
