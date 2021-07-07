# this is a Python BIDS converter

import glob
import os
import re
import pandas as pd
import numpy
#you can import this if you are making use of the dcm2bids module: import dcm2bids
import pydicom # ds = dcmread(file) if you want to obtain the dcm header
import shutil
from tkinter import filedialog
from tkinter import *
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

### ask for directory
#root = Tk()
#root.withdraw()
#folder_selected = filedialog.askdirectory()
#file=filedialog.askopenfilename()

### This is a data dictionary with the order in which the fixed entities need to be stored
BIDS_DICTS_Mapping = {0: 'sub-',
                      1: '_ses-',
                      2: '_task',
                      3: '_acq-',
                      4: '_ce-',
                      5: '_dir-',
                      6: '_rec-',
                      7: '_run-',
                      8: '_echo-',
                      9: '_',
                      10: '_mod-',
                      11: '_defacemask',
                      12: '_',
                      13: '_sbref'}

### reading in the maping excel sheet
BIDS_DICTS_Scanlabel = pd.read_excel(
    r'C:\\Users\\Jonathan\\Charité - Universitätsmedizin Berlin\\Interventional Cognitive Neuromodulation - '
    r'Data\\BIDS_conversionroom\\BIDS_Berlin_DBSlaboratory'
    r'\\Mapping_foldernames_to_BIDSlabels_with_scan_duration_no_doubles_jvh.xlsx',
    sheet_name='Python')
BIDS_DICTS_Scanlabel = BIDS_DICTS_Scanlabel.to_dict()

BIDS_DICTS_Scanlabel = {k.lower(): v
                        for k, v in
                        BIDS_DICTS_Scanlabel.items()
                        }

#location of the microGL location
DCM2NII='C:\\Users\\Jonathan\\Documents\\TOOLS\\MRIcroGL\\Resources\\dcm2niix.exe'

#location of the source folder with the dicom files

location_sourcedata= "C:\\Users\\Jonathan\\Documents\\DATA\\PROJECT_Berlin_DBSlaboratory"
#location_sourcedata= "C:\\Users\\Jonathan\\Charité - Universitätsmedizin Berlin\\Interventional Cognitive
# Neuromodulation - " \
#             "Data\\\BIDS_conversionroom\\BIDS_Berlin_DBSlaboratory\\sourcedata\\sub-001" #\
            # "\\KOPF_NEURO_KINDERMRT_DBS_VON_DK_20210116_103934_448000"
#location_sourcedata="C:\\Users\\Jonathan\\Documents\\DATA\\PROJECT_ALSP\\COLLABORATION_ALSP_Tuebingen\\HDLS_08-HDLS_AT_04"
print("The source folder is:")
print(location_sourcedata)

#location of the output folder
BIDS_MAIN=location_sourcedata
#BIDS_MAIN="C:\\Users\\Jonathan\\Charité - Universitätsmedizin Berlin\\Interventional Cognitive Neuromodulation - " \
#          "Data\\BIDS_conversionroom\\BIDS_Berlin_DBSlaboratory\\rawdata"
print("The output BIDS_MAIN folder is:")
print(BIDS_MAIN)

if not os.path.exists(BIDS_MAIN):
    os.makedirs(BIDS_MAIN)
    print("Directory '% s' created" % BIDS_MAIN)
else:
    print("Directory '% s' already exists" % BIDS_MAIN)
#keep track of what is not in csv file
not_in_dictionary=set()

#get the subjectdirs
Subjectdirs = glob.glob(location_sourcedata + os.sep + "*" + os.sep)
print("Dirs of subjects\n", *Subjectdirs, sep = "\n")

# iterate over the subjects
for Subjectdir in Subjectdirs:
    Subjectdir = os.path.basename(os.path.normpath(Subjectdir))
    #
    # Subject = re.search('(?<=_)(([0-9])*)', Subjectdir)
    #
    # Subject = "{:03d}".format(int(Subject.group(0)))
    # Subject = "sub-" + str(Subject)
    # Subject += '-AT' if 'HDLSAT' in Subjectdir else ''
    Subject = "sub-001"
    BIDS_Subjectdir = BIDS_MAIN + os.sep + Subject

    if not os.path.exists(BIDS_Subjectdir):
        os.makedirs(BIDS_Subjectdir)
        print("Directory '% s' created" % BIDS_Subjectdir)
    else:
        print("Directory '% s' already exists" % BIDS_Subjectdir)

    Sessiondirs = glob.glob(location_sourcedata + os.sep + Subjectdir + os.sep + "*" + os.sep)
    print("Dirs of sessions\n", *Sessiondirs, sep="\n")

    for i, Sessiondir in enumerate(Sessiondirs):
        # Sessionnumber = "{:03d}".format(i + 1);
        # Sessiondir = os.path.basename(os.path.normpath(Sessiondir))
        # Session = re.search('(?<=u)(([0-9.])*)', Sessiondir)
        # Session = Session.group(0)
        # Session = Session.replace('.', '')
        # Session = "ses-" + Sessionnumber + '-' + Session
        Session = "ses-Imaging01"
        BIDS_Sessiondir = BIDS_Subjectdir + os.sep + Session

        if not os.path.exists(BIDS_Sessiondir):
            os.makedirs(BIDS_Sessiondir)
            print("Directory '% s' created" % BIDS_Sessiondir)
        else:
            print("Directory '% s' already exists" % BIDS_Sessiondir)

        Scandirs = glob.glob(location_sourcedata + os.sep + Subjectdir + os.sep + Sessiondir + os.sep + "*" + os.sep)
        print("Dirs of scans\n", *Scandirs, sep="\n")
        if not Scandirs:
            Scandirs = Sessiondirs
        for Scandir in Scandirs:

            ###### CONVERT SCANDIR of SOURCE to BIDS_SCAN
            #### Start with subject and session


            Scandirfullname = Scandir[:-1]
            
            ### this is the MAIN operation. Convertion of Dicom to Nifti using the commandline dcm2niix.exe
            NIFTIcmd = DCM2NII + ' -f ' + '\"' + Subject + '_' + Session + '__%p' + '\"' + ' -p y -z n -o ' + '\"' + \
                       BIDS_Subjectdir + '\" \"' + Scandirfullname
            NIFTIcmd

            #### here create nifti with this command
            os.system(NIFTIcmd)
            print("Convert to nifti:")
            print(NIFTIcmd)

            #look for how many files were created
            Niftiscreated = glob.glob(BIDS_Subjectdir + os.sep + "*.nii")
            print("Nifties created:\n", *Niftiscreated, sep="\n")
            
            # the protocoll name is written out, now we want to change the file names into the different BIDS labels using the dictionary.
            #read the files created
            for Nifti in Niftiscreated:
                #save the full name
                Nifti_full = Nifti
                #remove directory name and the extension
                Nifti = os.path.splitext(os.path.basename(Nifti))[0]


                #get the %p of the dcm2nii conversion
                protocolname_full = re.search('(?<=__)(.*)', Nifti)
                protocolname_full = protocolname_full.group(0)
                #remove _ph from certain converstions
                protocolname_to_be_matched = protocolname_full.rsplit("_ph")
                protocolname_to_be_matched = protocolname_to_be_matched[0]
                protocolname_to_be_matched = protocolname_full.rsplit("_pha")
                protocolname_to_be_matched = protocolname_to_be_matched[0]
                #remove _e1 _e2 or _i0001 in certain conversions
                protocolname_to_be_matched = protocolname_to_be_matched.rsplit("_e0")
                protocolname_to_be_matched = protocolname_to_be_matched[0]
                protocolname_to_be_matched = protocolname_to_be_matched.rsplit("_e1")
                protocolname_to_be_matched = protocolname_to_be_matched[0]
                protocolname_to_be_matched = protocolname_to_be_matched.rsplit("_e2")
                protocolname_to_be_matched = protocolname_to_be_matched[0]
                protocolname_to_be_matched = protocolname_to_be_matched.rsplit("_e3")
                protocolname_to_be_matched = protocolname_to_be_matched[0]
                protocolname_to_be_matched = protocolname_to_be_matched.rsplit("_e4")
                protocolname_to_be_matched = protocolname_to_be_matched[0]
                protocolname_to_be_matched = protocolname_to_be_matched.rsplit("_e5")
                protocolname_to_be_matched = protocolname_to_be_matched[0]
                protocolname_to_be_matched = protocolname_to_be_matched.rsplit("_e6")
                protocolname_to_be_matched = protocolname_to_be_matched[0]
                protocolname_to_be_matched = protocolname_to_be_matched.rsplit("_e7")
                protocolname_to_be_matched = protocolname_to_be_matched[0]
                protocolname_to_be_matched = protocolname_to_be_matched.rsplit("_e8")
                protocolname_to_be_matched = protocolname_to_be_matched[0]
                protocolname_to_be_matched = protocolname_to_be_matched.rsplit("_e9")
                protocolname_to_be_matched = protocolname_to_be_matched[0]
                protocolname_to_be_matched = protocolname_to_be_matched.rsplit("_i0")
                protocolname_to_be_matched = protocolname_to_be_matched[0]
                protocolname_to_be_matched = protocolname_to_be_matched.lower() # make lowercase
                protocolname_to_be_matched_copy = protocolname_to_be_matched
                #### Look-up in data dictionary
                if not protocolname_to_be_matched in BIDS_DICTS_Scanlabel.keys():
                    protocolname_to_be_matched = protocolname_to_be_matched[:-1] #if protocolname ends on a b c
                if not protocolname_to_be_matched in BIDS_DICTS_Scanlabel.keys():
                    protocolname_to_be_matched = protocolname_to_be_matched[:-1]  # if protocolname ends _1 e.g. haste_localizer_I-III_1_i00005.nii
                if not protocolname_to_be_matched in BIDS_DICTS_Scanlabel.keys():
                    protocolname_to_be_matched = protocolname_to_be_matched[:-1]  # if protocolname ends _1 e.g. haste_localizer_I-III_1a_i00005.nii
                if protocolname_to_be_matched in BIDS_DICTS_Scanlabel.keys():

                    # ii = 0 is the subject
                    # ii = 1 is the session
                    # ii = 2-12 are the other labels stored in BIDS_DICTS_mapping, see line 23 of this code


                    ### determine the anat dwi swi or spine folder

                    if BIDS_DICTS_Scanlabel[protocolname_to_be_matched][9] == 'dwi':
                        BIDS_Scandir = BIDS_Sessiondir + os.sep + 'dwi'  # change to DWI only if label is present
                    elif BIDS_DICTS_Scanlabel[protocolname_to_be_matched][9] == 'swi':
                        BIDS_Scandir = BIDS_Sessiondir + os.sep + 'swi'  # change to SWI only if label is present
                    elif BIDS_DICTS_Scanlabel[protocolname_to_be_matched][9] == 'spine': ### spine is not a BIDS standard
                        BIDS_Scandir = BIDS_Sessiondir + os.sep + 'spine'  # change to SWI only if label is present
                    else:
                        BIDS_Scandir = BIDS_Sessiondir + os.sep + 'anat'  # change to DWI only if label is present

                    if not os.path.exists(BIDS_Scandir):
                        os.makedirs(BIDS_Scandir)
                        print("Directory '% s' created" % BIDS_Scandir)
                    else:
                        print("Directory '% s' already exists" % BIDS_Scandir)

                    ### check if another file with same name is already in the final folder -> runnumber increasing

                    niftiname_already_exists = True
                    runnumber = 1
                    BIDS_Scan = Subject + '_' + Session  # reset the Name
                    while niftiname_already_exists:
                        for ii in range(2, 12):

                            # if csv cell is not NaN
                            if not pd.isna(BIDS_DICTS_Scanlabel[protocolname_to_be_matched][ii]):

                                # add mapping and the label from excel sheet

                                BIDS_Scan += BIDS_DICTS_Mapping[ii]
                                BIDS_Scan += BIDS_DICTS_Scanlabel[protocolname_to_be_matched][ii]
                            # determine whether a run label need to be added

                            elif (ii == 7) and (runnumber > 1):
                                BIDS_Scan += BIDS_DICTS_Mapping[ii]
                                BIDS_Scan += str(runnumber)

                        ##### check if file already exists
                        if runnumber == 999:
                            error_stuck_in_while_loop

                        ##### check if file already exists

                        if os.path.isfile(os.path.normpath(BIDS_Scandir + os.sep + BIDS_Scan + ".nii")):
                            runnumber += 1
                            print('runnumber + 1')
                            BIDS_Scan = Subject + '_' + Session  # reset the Name
                        else:
                            niftiname_already_exists = False



                    #### for all nifti in niftiscreated move and rename the file (also .json files)
                    shutil.move(Nifti_full, BIDS_Scandir + os.sep + BIDS_Scan + ".nii")
                    print('moving and renaming ' + Nifti_full + ' to ' + BIDS_Scandir + os.sep + BIDS_Scan + ".nii")
                    # get corresponding json bval bvec file to the nifti file
                    for otherfile in glob.glob(BIDS_Subjectdir + os.sep + Nifti + ".*"):
                        shutil.move(otherfile, BIDS_Scandir + os.sep + BIDS_Scan + os.path.splitext(otherfile)[1])

                else:
                    print(protocolname_to_be_matched_copy + ' is not in dictionary')
                    not_in_dictionary.add(protocolname_to_be_matched_copy)
print("not in dictionary \n", *not_in_dictionary, sep="\n")
