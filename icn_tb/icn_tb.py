import json
import os
import pathlib
from glob import glob
import fileinput
import numpy as np
import psutil
from matplotlib import pyplot as plt



def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
            

def ci(srchlist, strlist):
    if isinstance(srchlist, type('a')):
        srchlist = [srchlist]
    i = list()
    for a in srchlist:
        i.extend(list(np.flatnonzero(np.char.startswith(strlist, a))))
    return i


def n_jobs():
    return int(psutil.cpu_count() - 1)


def fileparts(filename, append='', dropletters=None):
    fdir, fname = os.path.split(str(pathlib.Path(filename)))
    fname, ext = os.path.splitext(fname)
    if not append:
        return fdir, fname, ext
    else:
        if dropletters:
            fname = fname[:dropletters]
        return str(pathlib.Path(fdir, fname + append))


def ffind(folder='.', string='*.*'):
    files = []
    for cdir, _, _ in os.walk(folder):
        files.extend(glob(os.path.join(cdir, string)))
    return files


def replace_txt_in_file(filename,searchstring,replacestring=''):
        with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace(searchstring,replacestring), end='')


def json_write(filename, data):
    with open(pathlib.Path(filename).as_posix(), 'w') as json_file:
        json.dumps(data, indent=4)
        json.dump(data, json_file, indent=4)


def json_read(filename):
    with open(filename) as json_file:
        data = json.load(json_file)

    print(json.dumps(data, indent=4))
    return data


def status_write(filename, status_update):
    if not isinstance(status_update, type({})):
        fpath, fname = os.path.split(filename)
        status, ext = os.path.splitext(fname)
        status_update = {status: status_update}
    if not pathlib.Path.is_file(pathlib.Path(filename)):
        state = status_update
    else:
        state = json_read(pathlib.Path(filename))
        state.update(status_update)
    json_write(filename, state)


def status_check(filename, status=None, compare=None):
    filename = pathlib.Path(filename)
    if not status:
        fpath, fname = os.path.split(filename)
        status, ext = os.path.splitext(fname)

    if pathlib.Path.is_file(filename):
        state = json_read(filename)
        if isinstance(status, type('a')):
            status = [status]
        if len(status) >= 2:
            for a in status:
                state = state[a]
        if compare is not None:
            return state == compare
        else:
            return state
    else:
        print('not a file')
        return False


def plot(x, y=None, linewidth=1, color=None):
    if y is None:
        for a in np.arange(0, x.shape[0]):
            if color is None:
                plt.plot(x[a, :], linewidth=linewidth)
            else:
                plt.plot(x[a, :], linewidth=linewidth, color=color)
    else:
        for a in np.arange(0, y.shape[0]):
            if color is None:
                plt.plot(x, y[a, :], linewidth=linewidth)
            else:
                plt.plot(x, y[a, :], linewidth=linewidth, color=color)


def mkdir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)