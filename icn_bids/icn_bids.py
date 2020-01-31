import pandas as pd
import pathlib
from datetime import datetime
import shutil
import icn_tb as tb


def bids_get_participants_tsv_filename(bids_folder):
    return str(pathlib.Path(bids_folder, 'participants.tsv'))


def bids_read_participants_tsv(bids_folder):
    return pd.read_csv(bids_get_participants_tsv_filename(bids_folder), delimiter='\t')


def bids_backup_participants_tsv(bids_folder):
    fdir, fname, ext = tb.fileparts(bids_get_participants_tsv_filename(bids_folder))
    shutil.copyfile(str(pathlib.Path(fdir, fname + ext)), str(pathlib.Path(fdir, fname + '_backup' + ext)))


def bids_write_participants_tsv(df, bids_folder):
    bids_backup_participants_tsv(bids_folder)
    df.to_csv(bids_get_participants_tsv_filename(bids_folder), sep='\t', index=False)


def bids_get_participant_id_from_filename(filename):
    fdir, fname, ext = tb.fileparts(filename)
    dirparts = pathlib.Path(fdir).parts
    for item in list(dirparts):
        if item.startswith('sub-'):
            return item


def bids_get_participants(bids_folder, item='participant_id'):
    df = bids_read_participants_tsv(bids_folder)
    return list(df[item])


def bids_write_json_to_participants_tsv(bids_folder, json_file, participant_id=None):
    if participant_id is None:
        participant_id = bids_get_participant_id_from_filename(json_file)
    js = pd.read_json(json_file, typ='series')
    subs = bids_read_participants_tsv(bids_folder)
    i = [i for i, x in enumerate(subs.participant_id == participant_id) if x]
    for k in js.keys():
        print(k)
        print(str(js[k]))
        if js[k]:
            subs.loc[i, k] = str(js[k])
    subs.to_csv(bids_get_participants_tsv_filename(bids_folder), sep='\t', index=False)
    fdir, fname, ext = tb.fileparts(str(pathlib.Path(bids_folder, 'participants.json')))
    shutil.copyfile(pathlib.Path(fdir, fname + ext), pathlib.Path(fdir, fname + '_backup' + ext))
    js = tb.json_read(pathlib.Path(fdir, fname + ext))
    for a in subs.keys():
        if a not in list(js.keys()):
            js.update({a: {'Description': a}})
    tb.json_write(pathlib.Path(fdir, fname + ext), js)
    return subs


def date_string():
    return f'{datetime.now():%Y_%m_%d_%H_%M_%S%z}'

