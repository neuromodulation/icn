import IO
import os
import mne_bids
from mne_bids import update_sidecar_json # append this to __init__.py of the mne bids module from https://mne.tools/mne-bids/stable/_modules/mne_bids/sidecar_updates.html#update_sidecar_json
import mne
import pandas
import json as js
from datetime import datetime
from datetime import timezone
import xl2dict
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import csv
from tempfile import NamedTemporaryFile
from shutil import move

def get_all_paths(BIDS_path,extension):
    """

    Given a BIDS path return all vhdr file paths without BIDS_Layout
    Args:
        BIDS_path (string)
    Returns:
        vhdr_files (list)
    """
    paths = []
    for root, dirs, files in os.walk(BIDS_path):
        for file in files:
            if file.endswith(extension):
                paths.append(os.path.join(root, file))
    return paths

root=r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\BIDS_Berlin_ECOG_LFP\rawdata"
excelfile=r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\_non_Datasets\Documents\BIDS_Berlin_LFP_ECOG_templates\BIDS_Berlin_conversion_20210420.xls"
#make list of dictionary with list index the row number, and the dictionary has key:value pair as columnname:data
list_of_dicts= xl2dict.XlToDict()
list_of_dicts=list_of_dicts.convert_sheet_to_dict(file_path=excelfile)

#search for all scans.tsv
tsv_paths = get_all_paths(root, "scans.tsv")
print(*tsv_paths, sep="\n")

for tsv_path in tsv_paths:

    # names of files to read from
    r_filenameTSV = tsv_path

    # names of files to write to
    w_filenameTSV = tsv_path

    # read the data
    tsv_read = pd.read_csv(r_filenameTSV, sep='\t')
    #tsv_read = tsv_read.replace('\n', ' ', regex=True)
    #tsv_read = tsv_read.replace('nan', 'n/a', regex=True)
    try:
        #for every input of acq time, we search for the file name in excel. and update the acquisition time
        for i in range(len(tsv_read.acq_time)):
            searchfor=tsv_read.filename[i]
            searchfor = searchfor[5:-10]
            current_row_dict = next((item for item in list_of_dicts if item["Filename_BIDS"] == searchfor), None)
            bids_file = current_row_dict["Filename_BIDS"]

            try:
                dateobject=datetime.strptime(current_row_dict["acq_time"],'%Y-%m-%dT%H:%M:%S')
                dateobject = dateobject.replace(tzinfo=timezone.utc)
                tsv_read.acq_time[i] = dateobject.strftime('%Y-%m-%dT%H:%M:%S')

            except:
                print("has no acquitision date: ", bids_file)


        #the file is overwritten, but only if the subject was listed in the excel file.
        with open(w_filenameTSV, 'w', newline='') as write_tsv:
            write_tsv.write(tsv_read.to_csv(sep='\t', encoding='utf-8',  index=False, line_terminator='\r\n', na_rep='n/a'))
    except:
        print("subject has not been found: ", tsv_path)
