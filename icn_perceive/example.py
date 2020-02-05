# Example how files can be converted from a folder that contains only json reports from one subject

import icn_tb as tb
import icn_perceive as perceive
from icn_perceive import * # for debugging

root = 'Z:/LFP/PROJECTS/PERCEPT/' # this is where the subfolder containing the json files is located
bids_folder = 'Z:/LFP/PROJECTS/PERCEPT/BIDS' # this is the folder where the files should be exported to
subject = 'sub-002' # this is the subject id for subfolder and filename creation
files = tb.ffind(tb.pathlib.Path(root, subject), 'Report_*.json') # this searches for all the Reports in root/subject
filename = files[-1] # this defines a filename for convenience in testing and debugging
# Now just run through all the files and export all power spectra as tsv and time series as edf
for filename in files:
    filename = perceive.anonymize(filename)
    perceive.convert_to_bids(filename, subject, bids_folder)
