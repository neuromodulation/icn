import mne
import pybv
import mne_bids
import os

def fieldtrip_to_BIDS(filename, BIDS_path, subject_id, session, task, run):
    """
    Write BIDS entry from a fieldtrip format
    The function has to temporarily save the output file as brainvision to set the filenames attribute in the mne RawArray 
    Args:
        filename (string): .mat fieldtrip name
        BIDS_path (string): BIDS_path
        subject_id (string): BIDS subject id
        session (string): BIDS session
        task (string): BIDS session
        run (string): BIDS session
    """
    
    raw = mne.io.read_raw_fieldtrip(filename, None)
    ieegdata = raw.get_data()
    info = mne.create_info(raw.ch_names, raw.info['sfreq'], ch_types='ecog')
    raw = mne.io.RawArray(ieegdata, info)
    
    bids_basename = mne_bids.make_bids_basename(subject=subject_id, session=session, task=task, run=run)
    
    pybv.write_brainvision(raw.get_data(), raw.info['sfreq'], raw.ch_names, 'dummy_write', os.getcwd())
        
    bv_raw = mne.io.read_raw_brainvision('dummy_write.vhdr')
    
    # set all channel types to ECOG for iEEG - BIDS does not allow more than one channel type
    mapping = {}
    for ch in range(len(bv_raw.info['ch_names'])):
        mapping[bv_raw.info['ch_names'][ch]] = 'ecog'
    bv_raw.set_channel_types(mapping)
    
    mne_bids.write_raw_bids(bv_raw, bids_basename, BIDS_path, overwrite=True)

    #  remove dummy file
    os.remove('dummy_write.vhdr')
    os.remove('dummy_write.eeg')
    os.remove('dummy_write.vmrk')
    
    