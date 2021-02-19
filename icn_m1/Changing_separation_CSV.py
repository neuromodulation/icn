import os
import csv
#this is a code that will read in semicolon separated files and change them into comma separated files
# by Jonathan Vanhoecke
# 02/19/2021
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

inputdir=r'C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Jonathan\csv_input'
outputdir=r'C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Jonathan\csv_output'

csvpaths=get_all_paths(inputdir,'.csv')
for csvpath in csvpaths:
    csvfilename=os.path.basename(csvpath)
    with open(csvpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';') #this is the moment that the ; csv file are readed in
        with open(outputdir + os.sep + csvfilename , mode="w", newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',') #this is the moment that the , cvs is written out CAUTION: this assumes that there are NO other , in the files, such as comments or in strings
            try:
                writer.writerows(csv_reader)
            except:
                print('this file ' + csvfilename + ' has a comma emnbedded in a string, such as a comment and can not be converted or an non UTF decoded sign')