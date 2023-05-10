'''
this is a tool to relabel any file, filecontent or directory based on regular expressions
made by Jonathan Vanhoecke for the Neumann ICN Lab
01/01/2021
last update: 17.02.2022
'''
literal=False
doyouwanttkinter=True
#get target folder
#path_target_folder=r"C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\conversion_room\rawdata"
#path_target_folder=r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\BIDS_Tuebingen"
#path_target_folder=r"C:\Users\Jonathan\Documents\DATA\PROJECT_Tiantan\conversion_room\electrodes"
path_target_folder=r"C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\BIDS_01_Berlin_Neurophys\rawdata17"
path_target_folder=r"E:\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT MEG Connectomics\3_TUTTI"
back_up_folder=r"C:\Users\Jonathan\Documents\DATA\PROJECT_Berlin_dev\backup"
back_up_folder=r"E:\"

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from datetime import datetime
from functools import partial
import glob
import os
from os.path import splitext
import re
import pandas as pd
import numpy
import pathlib
import shutil
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import time
from functools import partial
import tkinter as tk
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from datetime import datetime
from functools import partial
import webbrowser


#################### INITIATE VARIABLES

ext_excluded = set()
dirs_excluded =set()
# what are you looking for?
#target_expression = "ses-001"
#change_into = "ses-20171014"



######################################### OPTEN TKINTER PART 1 ###########################################
###########################################################################################################

def browse_button():
    global path_target_folder
    path_target_folder = filedialog.askdirectory()
    global lb_22
    lb_22.config(text=path_target_folder)

def callback(url):

    webbrowser.open_new(url)

go1=False
def setgo1():
    global go1
    go1=True
    search()
    global target_expression_list, change_into_list
    target_expression_list=entry_target_expression.get("1.0", tk.END)
    target_expression_list=target_expression_list.split("\n")
    change_into_list = entry_change_into.get("1.0", tk.END)
    change_into_list = change_into_list.split("\n")
    target_expression_list[:] = [x for x in target_expression_list if x]
    change_into_list[:] = [x for x in change_into_list if x]


    window.destroy()

    if len(target_expression_list)!=len(change_into_list):
        if len(target_expression_list)==1:
            change_into_list=[''] # which means you will remove the target expression
        else:
            error("len target_expression must be equal to len change into")





def confirm_checkboxes():


        if 'cbs_ext' in globals():
            global ext_excluded
            for i, checks in enumerate(cbs_ext):
                if checks.instate(['!selected']):
                    ext_excluded.add(ext_found[i])

        if 'cbs_subfolders' in globals():
            global dirs_excluded
            for i, checks in enumerate(cbs_subfolders):
                if checks.instate(['!selected']):
                    dirs_excluded.add(list_of_paths_subfolders[i])

def search():

    confirm_checkboxes()
    global search_subfolders_boolean, list_of_paths_subfolders, list_of_paths_files, ext_found

    search_subfolders_boolean = chk_include_subfolders.instate(['selected'])

    list_of_paths_subfolders, list_of_paths_files, ext_found = list_dirs_and_files(path_target_folder, ext_excluded,
                                                                                   dirs_excluded,
                                                                                   search_subfolders_boolean)



    global Frame_ext

    lb_8 = tk.Label(text="included filetypes: ", master=frame, bg="deep pink",
                    anchor="e")
    lb_8.grid(row=8, column=1, sticky=E + W, padx=5, pady=5)

    Frame_ext = tk.Frame(master=frame, bg="blue")
    Frame_ext.grid(row=8, column=2, padx=5, pady=5, sticky=W)

    #button_confirm = tk.Button(master=Frame_ext, text="Confirm selection to search my regex",
                               # command= partical(blabalbal) #
                               #command=confirm_checkboxes)  # print(3))#(
    # root).confirm_checkboxes(3))
    #button_confirm.pack(padx=5, pady=5, side="left")

    vsb = tk.Scrollbar(Frame_ext, orient="vertical")
    text = tk.Text(Frame_ext, height=15, width=150,
                   yscrollcommand=vsb.set)
    vsb.config(command=text.yview)
    vsb.pack(side="left", fill="y")
    text.pack(side="left", fill="both")#, expand=True)

    #vars = []
    global cbs_ext
    cbs_ext = []

    for i, ext in enumerate(ext_found):
        #var = tk.IntVar()

        cb = ttk.Checkbutton(Frame_ext, text="file%s" % ext)  # ,onvalue=1,offvalue=0)

        cb.state(['!alternate'])
        cb.state(['selected'])

        text.window_create("end", window=cb)
        text.insert("end", "\n")  # to force one checkbox per line
        #vars.append(var)
        cbs_ext.append(cb)
    text.config(state="disabled")
    #### seach in subfolders

    if search_subfolders_boolean:

        global Frame_subfolders

        lb_9 = tk.Label(text="included subfolders: ", master=frame, bg="deep pink",
                        anchor="e")
        lb_9.grid(row=9, column=1, sticky=E + W, padx=5, pady=5)

        Frame_subfolders = tk.Frame(master=frame, bg="black")
        Frame_subfolders.grid(row=9, column=2, padx=5, pady=5, sticky=W)

        #button_confirm = tk.Button(master=Frame_ext, text="Confirm selection to search my regex",
         #                          # command= partical(blabalbal) #
          #                         command=confirm_checkboxes)  # print(3))#(
        # root).confirm_checkboxes(3))
        #button_confirm.pack(padx=5, pady=5, side="left")

        vsb2 = tk.Scrollbar(Frame_subfolders, orient="vertical")
        vsb3 = tk.Scrollbar(Frame_subfolders, orient="horizontal")
        text2 = tk.Text(Frame_subfolders, height=15, width=150,
                       yscrollcommand=vsb2.set, xscrollcommand=vsb3.set)
        vsb2.config(command=text2.yview)
        vsb2.pack(side="left", fill="y")


        vsb3.config(command=text2.xview)
        vsb3.pack(side="bottom", fill="x")
        text2.pack(side="left", fill="both")#, expand=True)

        global cbs_subfolders
        cbs_subfolders = []

        for i, subfolder in enumerate(list_of_paths_subfolders):
            #var = tk.IntVar()

            cb = ttk.Checkbutton(Frame_subfolders, text="%s " % subfolder)  # ,onvalue=1,offvalue=0)

            cb.state(['!alternate'])
            cb.state(['selected'])

            text2.window_create("end", window=cb)
            text2.insert("end", "\n")  # to force one checkbox per line
            #vars.append(var)
            cbs_subfolders.append(cb)
        text2.config(state="disabled")


# how to find all files and dirs regardless of what you are looking for but filtered on extension
def splitext_strong(path):
    return ''.join(pathlib.Path(path).suffixes)


def list_dirs_and_files(dirs, ext_excluded, dirs_excluded, search_subfolders_boolean=True):  # dir: str, ext: set
    list_of_paths_subfolders, list_of_paths_files, ext_found = [], [], []
    ext_found = set()
    for f in os.scandir(dirs):
        if f.is_dir() and search_subfolders_boolean:
            fpath = f.path
            if fpath not in dirs_excluded:
                list_of_paths_subfolders.append(fpath)
        if f.is_file():
            current_ext = str(splitext_strong(f.name))
            print(current_ext)
            current_ext = current_ext.lower()
            if current_ext not in ext_excluded:  # check whether you get the right extension in nii.gz
                # files
                list_of_paths_files.append(f.path)
                ext_found.add(current_ext)
    if search_subfolders_boolean:
        for dirs in list(list_of_paths_subfolders):
            if dirs not in dirs_excluded:
                sf, f, exs = list_dirs_and_files(dirs, ext_excluded, dirs_excluded)
                list_of_paths_subfolders.extend(sf)
                list_of_paths_files.extend(f)
                ext_found.update(exs)
    return list_of_paths_subfolders, list_of_paths_files, list(ext_found)


def onFrameConfigure(canvas): # https://stackoverflow.com/questions/3085696/adding-a-scrollbar-to-a-group-of-widgets-in-tkinter/3092341#3092341
    '''Reset the scroll region to encompass the inner frame'''
    canvas.configure(scrollregion=canvas.bbox("all"))

window = tk.Tk()
window.geometry("1230x620")
window.title("Relabelling Tool - ICN Neumann Lab")
window.resizable(True, True)
#path_target_folder="none"

canvas = tk.Canvas(window, borderwidth=0, background="black")
frame = tk.Frame(canvas, background="black")
vsb = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=vsb.set)
vsb.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)
canvas.create_window((4,4), window=frame, anchor="nw")
frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))



lb_1 = tk.Label(text="select the folder: ", master=frame, bg="deep pink", anchor="e")
lb_1.grid(row=1,column=1, sticky=E+W,  padx=5, pady=5)

button_browse = tk.Button(master=frame, text="Browse folder", command=browse_button)
button_browse.grid(row=1 , column=2, sticky="w", padx=5, pady=5)

lb_2 = tk.Label(text="selected folder: ", master=frame, bg="deep pink", anchor="e")
lb_2.grid(row=2,column=1, sticky=E+W,  padx=5, pady=5)

lb_22 = tk.Label(text=path_target_folder, master=frame, bg="white", anchor="w")
lb_22.grid(row=2,column=2, sticky=E+W,  padx=5, pady=5)

lb_3 = tk.Label(text="find regular expression: ", master=frame, bg="deep pink",
                anchor="e")
lb_3.grid(row=3,column=1, sticky=E+W,  padx=5, pady=5)

entry_target_expression = ScrolledText(master=frame, width=100, background='white')
entry_target_expression.grid(row=3 , column=2, sticky="w", padx=5, pady=5)
#entry_target_expression.insert("e.g. ses-[0-9]*")

lb_4 = tk.Label(text="change match into: ", master=frame, bg="deep pink", anchor="e")
lb_4.grid(row=4,column=1, sticky=E+W,  padx=5, pady=5)

entry_change_into = ScrolledText(master=frame, width=100, background='green')
entry_change_into.grid(row=4, column=2, sticky="w", padx=5, pady=5)
#entry_change_into.insert(0,"e.g. session-ecog")

lb_5 = tk.Label(text="further information: ", master=frame, bg="deep pink", anchor="e")
lb_5.grid(row=5,column=1, sticky=E+W,  padx=5, pady=5)

link1 = Label(text="click here to test my regexr", master=frame, fg="white", bg="black", cursor="hand2", anchor="w")
link1.grid(row=5,column=2, sticky=E+W)
link1.bind("<Button-1>", lambda e: callback("https://regexr.com/"))

lb_6 = tk.Label(text="search for subfolders: ", master=frame, bg="deep pink", anchor="e")
lb_6.grid(row=6,column=1, sticky=E+W,  padx=5, pady=5)

chk_include_subfolders = ttk.Checkbutton(master=frame)
chk_include_subfolders.grid(row=6, column=2, sticky=W, padx=5, pady=5)
chk_include_subfolders.state(['!alternate'])
chk_include_subfolders.state(['selected'])

lb_7 = tk.Label(text="determine filetypes and/or subfolders: ", master=frame,
                bg="deep pink",
                anchor="e")
lb_7.grid(row=7, column=1, sticky=E+W,  padx=5, pady=5)

button_search = tk.Button(master=frame, text="Explore current selection/ update my selection", command=search)
button_search.grid(row=7 , column=2, sticky="w", padx=5, pady=5)

button_confirm = tk.Button(master=frame, text="Save selection and search my regex",
                                      command=setgo1)
button_confirm.grid(row=10, column=2, sticky="w", padx=5, pady=5)


#######


window.mainloop()

# confirm your chose for regex into changes
window = tk.Tk()
for i in range(len(target_expression_list)):  # Rows

    b = tk.Label(window,text=target_expression_list[i])
    b.grid(row=i, column=1)
    b = tk.Label(window,text=change_into_list[i])
    b.grid(row=i, column=2)

window.mainloop()










if go1:
    #######################################################################################
    ####################################### MAIN CODE #####
    #######################################################################################
    # create back-up folder
    now = datetime.now()
    dir_names_changed = back_up_folder + os.sep + "names_changed_at_" + now.strftime(
        "%d-%m-%Y_time_%H-%M-%S")
    os.mkdir(dir_names_changed)
    logfile = open(dir_names_changed + os.sep + "log.txt", "a")
    logfile.write("log for " + dir_names_changed + "\n")

    for i in range(len(target_expression_list)):
        target_expression=target_expression_list[i]
        change_into=change_into_list[i]
        # create the pattern object. Note the "r". In case you're unfamiliar with Python
        # this is to set the string as raw so we don't have to escape our escape characters
        # to escape characters begin target expression with r such as r'(?<=abc)def'
        #if "\\" in r"%r" % target_expression:
        #    pattern=re.compile(r'{}'.format(target_expression.replace('\\','\\\\'))) #the target expression with
        #    always be treated as raw string
        #    change_into=r'{}'.format(change_into.replace('\\', '\\\\'))
        #else:
        if literal:
            pattern=re.compile(r'{}'.format(target_expression)) # take it literally
        else:
            pattern = re.compile(target_expression) # use a regex
        list_of_paths_subfolders, list_of_paths_files, ext_found = list_dirs_and_files(path_target_folder, ext_excluded,
                                                                                       dirs_excluded,
                                                                                       search_subfolders_boolean)


        # subfolders=glob.glob(target_folder + os.sep + "*" + os.sep)

        #### PART 1: Search
        #define output
        df_content = pd.DataFrame(columns=('path_current_file', 'current_file', 'change_lines_number', 'change_lines',
                                   'change_lines_into','change_matches'))
        df_filenames = pd.DataFrame(columns=('path_current_file','current_file','change_filename_into'))
        df_dirnames = pd.DataFrame(columns=('path_subfolders_to_be_changed','change_dirnames_into'))



        #path_current_file = os.path.normpath(list_of_paths_files[0])
        #current_file = os.path.basename(path_current_file)
        #current_file_ext = splitext_(current_file)



        ##### SEARCH IN THE FILE CONTENT AND NAME

        for n in range(0,len(list_of_paths_files)):
            # open files and search
            path_current_file = os.path.normpath(list_of_paths_files[n])
            current_file = os.path.basename(path_current_file)
            current_file_ext = splitext_strong(current_file)

            # check if files is above 10MB or ext is eeg or nii
            # check if files is above a certain size or ext is eeg or nii
            size_limit = 10485760
            if os.stat(path_current_file).st_size >= size_limit:
                print(
                    f"Excluded from reading content: the following file is more than "
                    f"{size_limit // (1024 * 1024)}MB in size: {current_file}."
                )
            elif current_file_ext in [".mat",".nii",".nii.gz",".eeg",".edf"]:
                print("Excluded from reading content: file extension is not accepted. " + current_file)
            else:
                print("Searching in file " + current_file)



                #match = pattern.search(current_file_content)

                #Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
                #path_current_file = filedialog.askopenfilename()  # show an "Open" dialog box and return the path to the
                # selected file
                print(path_current_file)



                ##### SEARCH IN THE FILE CONTENT
                try:
                    f_in = open(path_current_file, 'r')


                    change_lines=[]
                    change_lines_number=[]
                    change_matches=[]
                    change_lines_into=[]
                    change_filename=False
                    change_file_content=False

                    for i, line in enumerate(open(path_current_file, 'r')):
                        if re.findall(pattern, line):
                            print('Found on line %s: %s: %s' % (i+1, 'match', line)) #line number

                            # starts with
                            # 1, so no i +1
                            #print(match.start(), match.end()) #save the location
                            change_lines_number.append(i+1)
                            change_lines.append(line)
                            change_lines_into.append(re.sub(pattern, change_into, line))
                            #change_matches.append(match.span)

                            change_file_content=True
                    f_in.close()
                    if change_file_content:
                        df_content = df_content.append(pd.Series(), ignore_index=True)
                        df_content.iloc[-1]['path_current_file']=path_current_file
                        df_content.iloc[-1]['current_file'] = current_file
                        df_content.iloc[-1]['change_lines_number'] = change_lines_number
                        df_content.iloc[-1]['change_lines'] = change_lines
                        df_content.iloc[-1]['change_lines_into'] = change_lines_into
                        df_content.iloc[-1]['change_matches'] = change_matches

                except:
                    print("Excluded from reading: file cannot be read. " + current_file)





            ##### SEARCH IN THE FILE NAME
            change_filename=False
            if re.findall(pattern, current_file):
                change_filename=True
                change_filename_into=re.sub(pattern, change_into, current_file)
                df_filenames = df_filenames.append(pd.Series(), ignore_index=True)
                df_filenames.iloc[-1]['path_current_file'] = path_current_file
                df_filenames.iloc[-1]['current_file'] = current_file
                df_filenames.iloc[-1]['change_filename_into'] = change_filename_into
            elif re.findall(pattern, path_current_file):
                change_filename=True
                change_filename_into=re.sub(pattern, change_into, path_current_file)
                change_filename_into=os.path.basename(change_filename_into)
                df_filenames = df_filenames.append(pd.Series(), ignore_index=True)
                df_filenames.iloc[-1]['path_current_file'] = path_current_file
                df_filenames.iloc[-1]['current_file'] = current_file
                df_filenames.iloc[-1]['change_filename_into'] = change_filename_into


        ###### SEARCH IN THE DIR NAME

        change_dirnames_into = []
        path_subfolders_to_be_changed = []
        change_dirname = False
        for path_current_subfolder in list_of_paths_subfolders:
            current_subfolder_after_target_folder=path_current_subfolder[len(path_target_folder):]
            #current_dirname = os.path.basename(path_current_subfolder)
            if re.findall(pattern, current_subfolder_after_target_folder):
                new_dirname=re.sub(pattern, change_into,
                                   current_subfolder_after_target_folder) # be careful with escaping \\
                if os.path.dirname(new_dirname) == os.path.dirname(current_subfolder_after_target_folder):
                    change_dirnames_into.append(new_dirname)
                    path_subfolders_to_be_changed.append(path_current_subfolder)
                    print('Found on match in dir %s => %s' % (current_subfolder_after_target_folder, new_dirname))  # line number starts with

                    change_dirname = True

        if change_dirname:
            for i in range(0,len(path_subfolders_to_be_changed)):
                df_dirnames = df_dirnames.append(pd.Series(), ignore_index=True)
                df_dirnames.iloc[-1]['path_subfolders_to_be_changed'] = path_subfolders_to_be_changed[i]
                df_dirnames.iloc[-1]['change_dirnames_into'] = change_dirnames_into[i]
























        if doyouwanttkinter:
        ###############################################################################################################
        #### OPEN TKINTER PART 2 #############################################################################

            def do_not_change(r, entrytoken, var, lines=None):
                if var == 'content':

                    if lines is not None:
                        global df_content
                        df_content.iloc[r]['change_lines_number'][lines] = []
                        df_content.iloc[r]['change_lines'][lines] = []
                        df_content.iloc[r]['change_lines_into'][lines] = []
                elif var == 'filenames':

                    global df_filenames
                    df_filenames.iloc[r]['path_current_file'] = []
                    df_filenames.iloc[r]['current_file'] = []
                    df_filenames.iloc[r]['change_filename_into'] = []

                elif var == 'dirnames':

                    global df_dirnames
                    df_dirnames.iloc[r]['path_subfolders_to_be_changed'] = []
                    df_dirnames.iloc[r]['change_dirnames_into'] = []
                else:
                    raise ('we have an error with var type')
                entrytoken.insert(0, ' ')
                entrytoken.delete(0, 'end')
                entrytoken.insert(0, 'remains unchanged ')
                entrytoken['state'] = 'disabled'


            def save_my_changes(r, entrytoken, var, lines=None):

                if var == 'content':
                    if lines is not None:
                        global df_content
                        df_content.iloc[r]['change_lines_into'][lines] = entrytoken.get()
                elif var == 'filenames':
                    global df_filenames
                    df_filenames.iloc[r]['change_filename_into'] = entrytoken.get()

                elif var == 'dirnames':
                    global df_dirnames
                    df_dirnames.iloc[r]['change_dirnames_into'] = entrytoken.get()
                else:
                    raise ('we have an error with var type')

                entrytoken['background'] = 'green'
                entrytoken.config(validate='key', validatecommand=partial(change_color, entrytoken))


            def change_color(entrytoken):
                entrytoken['background'] = 'deep pink'


            go2 = False


            def setgo2():
                global go2
                global window
                go2 = True
                window.destroy()


            window = tk.Tk()
            window.geometry("1230x620")
            window.title("Relabelling Tool - ICN Neumann Lab")

            # Scrollbar addition from Codemy.com tutorial #96
            # create a main frame
            main_frame = Frame(window, bg="black")
            main_frame.pack(fill=BOTH, expand=True)

            # create the main canvas inside the main frame
            main_canvas = Canvas(main_frame, bg="black")
            main_canvas.pack(side=LEFT, fill=BOTH, expand=True)

            # add a scrollbar to the main canvas
            # the scroll bar is positioned in the main frame but attached to the canvas
            main_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=main_canvas.yview)
            main_scrollbar.pack(side=RIGHT, fill=Y)

            # configure the canvas
            # you need to bind the configuration to the scrollbar and determine the surface
            main_canvas.configure(yscrollcommand=main_scrollbar.set)
            main_canvas.bind('<Configure>', lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"), height=400))

            # create the second frame inside the canvas
            frame_second = Frame(master=main_canvas, bg="black")  # width=400, height=200, bg="black")
            # frame_second.pack(fill=BOTH, expand=True, sticky=E+W)
            # frame_second.grid(row=0, column=0, sticky=E+W)
            # add the second frame to a window in the canvas
            main_canvas.create_window((0, 0), window=frame_second, anchor="nw")

            # frame_second = tk.Frame(master=window, width=400, height=200, bg="black")
            # frame_second.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

            entries = []
            labels = []
            buttons = []
            row = -1
            for r in range(0, len(df_content.index)):
                var = 'content'
                row += 1
                lb_main = tk.Label(text=df_content.iloc[r]['current_file'], master=frame_second, bg="deep pink")
                lb_main.grid(row=row + 1, column=0)
                frame_sub = tk.Frame(master=frame_second, width=110, height=70, bg="gray")
                frame_sub.grid(row=row + 1, column=1, padx=5, pady=5)

                for lines in range(0, len(df_content.iloc[r]['change_lines'])):
                    lb_number = tk.Label(text=df_content.iloc[r]['change_lines_number'][lines], master=frame_sub, width=10,
                                         anchor="w")
                    lb_number.grid(row=lines * 2, column=0, padx=5, pady=5)
                    labels.append(lb_number)

                    lb_change_lines = tk.Label(text=df_content.iloc[r]['change_lines'][lines], master=frame_sub, width=100,
                                               anchor="w")
                    lb_change_lines.grid(row=lines * 2, column=1, padx=5, pady=5)
                    labels.append(lb_change_lines)

                    en = tk.Entry(master=frame_sub, width=100, background='green')
                    en.grid(row=lines * 2 + 1, column=1, sticky="w", padx=5, pady=5)

                    en.insert(0, df_content.iloc[r]['change_lines_into'][lines])
                    en.config(validate='key', validatecommand=partial(change_color, en))

                    entries.append(en)

                    btn_do_not_change = tk.Button(master=frame_sub, text="Do not change",
                                                  command=partial(do_not_change, r, en, var,
                                                                  lines))
                    btn_do_not_change.grid(row=lines * 2, column=2, padx=5, pady=5, sticky=E + W)
                    buttons.append(btn_do_not_change)
                    btn_save_my_changes = tk.Button(master=frame_sub, text="Save my changes",
                                                    command=partial(save_my_changes, r, en, var, lines),
                                                    activebackground='green')
                    btn_save_my_changes.grid(row=lines * 2 + 1, column=2, padx=5, pady=5)
                    buttons.append(btn_save_my_changes)

            for r in range(0, len(df_filenames.index)):
                var = 'filenames'
                row += 1
                lb_main = tk.Label(text=df_filenames.iloc[r]['current_file'], master=frame_second, bg="deep pink")
                lb_main.grid(row=row + 1, column=0)
                frame_sub = tk.Frame(master=frame_second, width=110, height=70, bg="black")
                frame_sub.grid(row=row + 1, column=1, padx=5, pady=5, sticky=E)

                lb_change_files = tk.Label(text=df_filenames.iloc[r]['current_file'], master=frame_sub, width=100,
                                           anchor="w")
                lb_change_files.grid(row=1, column=1, padx=5, pady=5)
                labels.append(lb_change_files)

                en = tk.Entry(master=frame_sub, width=100, background='green')
                en.grid(row=2, column=1, sticky="w", padx=5, pady=5)

                en.insert(0, df_filenames.iloc[r]['change_filename_into'])
                en.config(validate='key', validatecommand=partial(change_color, en))

                entries.append(en)

                btn_do_not_change = tk.Button(master=frame_sub, text="Do not change",
                                              command=partial(do_not_change, r, en, var))
                btn_do_not_change.grid(row=1, column=2, padx=5, pady=5, sticky=E + W)
                buttons.append(btn_do_not_change)
                btn_save_my_changes = tk.Button(master=frame_sub, text="Save my changes",
                                                command=partial(save_my_changes, r, en, var), activebackground='green')
                btn_save_my_changes.grid(row=2, column=2, padx=5, pady=5)
                buttons.append(btn_save_my_changes)

            for r in range(0, len(df_dirnames.index)):
                var = 'dirnames'
                row += 1
                lb_main = tk.Label(text=df_dirnames.iloc[r]['path_subfolders_to_be_changed'], master=frame_second,
                                   bg="deep pink")
                lb_main.grid(row=row + 1, column=0)
                frame_sub = tk.Frame(master=frame_second, width=110, height=70, bg="black")
                frame_sub.grid(row=row + 1, column=1, padx=5, pady=5, sticky=E)

                lb_change_files = tk.Label(text=df_dirnames.iloc[r]['path_subfolders_to_be_changed'],
                                           master=frame_sub,
                                           width=100,
                                           anchor="w")
                lb_change_files.grid(row=1, column=1, padx=5, pady=5)
                labels.append(lb_change_files)

                en = tk.Entry(master=frame_sub, width=100, background='green')
                en.grid(row=2, column=1, sticky="w", padx=5, pady=5)

                en.insert(0, df_dirnames.iloc[r]['change_dirnames_into'])
                en.config(validate='key', validatecommand=partial(change_color, en))

                entries.append(en)

                btn_do_not_change = tk.Button(master=frame_sub, text="Do not change",
                                              command=partial(do_not_change, r, en, var))
                btn_do_not_change.grid(row=1, column=2, padx=5, pady=5, sticky=E + W)
                buttons.append(btn_do_not_change)
                btn_save_my_changes = tk.Button(master=frame_sub, text="Save my changes",
                                                command=partial(save_my_changes, r, en, var), activebackground='green')
                btn_save_my_changes.grid(row=2, column=2, padx=5, pady=5)
                buttons.append(btn_save_my_changes)

            ### Final layout

            btn_exit_and_continue_with_changes = tk.Button(master=frame_second, text="Exit and continue with all changes",
                                                           command=setgo2)
            btn_exit_and_continue_with_changes.grid(row=row + 2, column=1, padx=5, pady=5, sticky='e')  # row +2 column 1
            buttons.append(btn_exit_and_continue_with_changes)


            window.mainloop()
        else:
            go2=True
















        ############################ MAIN PART 2 #####################################################
        ##################################################################################################

        if go2:


            #### PART 2
                # initiate to replace
                # start with making backup on the dir names changed location and open a new file
            for n in range(0,len(df_content.index)):
                if not df_content['path_current_file'][n]:
                    print('skip empty list')
                elif not df_content['change_lines_number'][n]:
                    print('skip empty list')
                else:
                    path_current_file = df_content['path_current_file'][n]
                    current_file = df_content['current_file'][n]
                    change_lines_number = df_content['change_lines_number'][n]
                    change_lines_into = df_content['change_lines_into'][n]

                    path_current_file_backup = os.path.normpath(dir_names_changed + os.sep + path_current_file[len(
                        path_target_folder):])


                    try:
                        shutil.move(path_current_file, path_current_file_backup)
                    except IOError as io_err:
                        os.makedirs(os.path.dirname(path_current_file_backup))
                        shutil.move(path_current_file, path_current_file_backup)

                    # check whether filename needs to be changed
                    #### start with adapting new filename if necessary
                    if df_filenames['current_file'].str.match(current_file).any():
                        index = df_filenames[df_filenames['current_file']==current_file].index.values.astype(int)[0]
                        change_filename_into = df_filenames['change_filename_into'][index]

                        logfile.write('Changed filename %s \n into %s \n' % (path_current_file, change_filename_into))
                        path_current_file = os.path.normpath(os.path.dirname(path_current_file) + os.sep +
                                                             change_filename_into)
                        df_filenames.iloc[index]['path_current_file'] = []
                        df_filenames.iloc[index]['current_file'] = []
                        df_filenames.iloc[index]['change_filename_into'] = []

                    f_in = open(path_current_file_backup, 'r')
                    f_out = open(path_current_file, 'x')
                    logfile.write('Changed content in file %s \n' % (path_current_file))
                    i = 1 # start by line number 1
                    for line in f_in:
                        if i in change_lines_number:
                            # replace the expression

                            f_out.write(change_lines_into[change_lines_number.index(i)])
                            logfile.write('Changed on line %d \n %s \n %s' % (i, line, change_lines_into[
                                change_lines_number.index(i)]))
                        else:
                            # write the line the way it is
                            f_out.write(line)
                        i += 1

                    f_in.close()
                    f_out.close()

            for n in range(0, len(df_filenames.index)):
                if not df_filenames['path_current_file'][n]:
                    print('skip empty list')
                else:
                    path_current_file = os.path.normpath(df_filenames['path_current_file'][n])
                    current_file = df_filenames['current_file'][n]
                    change_filename_into = df_filenames['change_filename_into'][n]

                    path_current_file_backup = os.path.normpath(dir_names_changed + os.sep + path_current_file[len(
                        path_target_folder):])

                    #shutil.move(path_current_file, path_current_file_backup)
                    try:
                        shutil.move(path_current_file, path_current_file_backup)
                    except IOError as io_err:
                        os.makedirs(os.path.dirname(path_current_file_backup))
                        shutil.move(path_current_file, path_current_file_backup)

                    # change the name by making a backup
                    logfile.write('Changed filename %s \n into %s \n' % (path_current_file, change_filename_into))

                    path_current_file = os.path.normpath(os.path.dirname(path_current_file) + os.sep + \
                                        change_filename_into)
                    shutil.copy(path_current_file_backup, path_current_file)


            for n in range(0, len(df_dirnames.index)):
                if not df_dirnames['path_subfolders_to_be_changed'][n]:
                    print('skip empty list')
                else:

                    path_subfolder_to_be_changed = df_dirnames['path_subfolders_to_be_changed'][n]
                    change_dirname_into = df_dirnames['change_dirnames_into'][n]
                    os.rename(path_subfolder_to_be_changed,path_target_folder + change_dirname_into) #os.sep ? in case of access error: PLEASE UNCHECK READ-ONLY properties in the folder
                    logfile.write('Changed dir %s \n into %s \n' % (path_subfolder_to_be_changed,
                                                                      path_target_folder + os.sep + change_dirname_into))

        # go2 is not TRUE
        else:
            print('Exit without changes')

# go1 is not TRUE
else:
    print('Exit without changes')



#### WRITE REPORT
