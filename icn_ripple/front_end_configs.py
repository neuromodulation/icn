"""
This script is collection of functions show examples of
how to use xipppy to interact with non-implantable front ends
connected to the NIP.
"""

import xipppy as xp

FS_CLK = 30000


def print_nip_info():
    """
    xipppy informational functions.
    """
    print('NIP serial number= ', xp.get_nip_serial())
    print('NIP exec version number= ', xp.get_nipexec_version())
    print('Connected front end version= ', xp.get_fe_version(0))


def print_min_elapsed():
    """
    This is an example shows definition of xipppy.time().
    """
    elapsed_time_seconds = xp.time() / FS_CLK
    elapsed_time_minutes = elapsed_time_seconds / 60
    print("Time elapsed after NIP start up: {:.2f} min ".
          format(elapsed_time_minutes))


def get_fe_first_electrode(front_end_type):
    """
    This function returns first electrode (index 0) of connected front_end.

    :param front_end_type: supported front end type
    :return: index 0 electrode of provided front end. None if not connected.
    """
    if (len(xp.list_elec(front_end_type)) > 0):
        return xp.list_elec(front_end_type)[0]
    else:
        return None


def fe_electrodes_and_streams(front_end_types):
    """
    xipppy.list_elec() and xipppy.get_fe_streams() usage:

    This example prints electrode numbers and type of supported streams for
    each front end that connected to the NIP.

    :param front_end_types: list of front ends types
    """

    all_elecs = []
    fe_elecs = []
    for fe_type in front_end_types:
        all_elecs.append(xp.list_elec(fe_type))
        if len(all_elecs[-1]) > 0:
            fe_elecs.append(all_elecs[-1][0])
            print('{:s} electrode numbers:'.format(fe_type))
            print(all_elecs[-1])
            print('{:s} streams:'.format(fe_type))
            print(xp.get_fe_streams(fe_elecs[-1]))
        else:
            fe_elecs.append(None)


def print_enabled_streams(front_end_types):
    """
    xipppy.get_fe(), xipppy.signal() usage:
    This example shows available streams for each connected front end.

    :param front_end_types: list of frond end types
    """
    streams = ['raw', 'lfp', 'hi-res', 'spk', 'stim']

    print('Enabled streams:')
    for i, fe_type in enumerate(front_end_types):
        fe_electrode = get_fe_first_electrode(fe_type)
        if fe_electrode is not None:
            print('{:s} FE(s) (indices {:s}):'.format(fe_type, str(
                xp.get_fe(fe_electrode))))
            for stream in streams:
                stream_is_active = []
                try:
                    stream_is_active.append(
                        str(xp.signal(fe_electrode, stream)))
                except xp.exception as e:
                    stream_is_active.append('N/A')
                print(
                    '{:s}:\t{:s}'.format(stream, ', '.join(stream_is_active)))


if __name__ == '__main__':
    with xp.xipppy_open():
        front_end_types = ['stim', 'micro', 'nano', 'surf', 'EMG', 'analog']
        # NIP and FE info
        print_nip_info()
        # Elapsed time
        print_min_elapsed()
        # Streaming types
        fe_electrodes_and_streams(front_end_types)
        # Fe type streams:
        print_enabled_streams(front_end_types)