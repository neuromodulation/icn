"""
This script is an example of how to get and process spike data using xipppy.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import empty

import xipppy as xp

SPK_SAMPLES = 52
FS_CLK = 30000
SPK_BUFFER_SIZE = 100


def _unit_color(class_id):
    return {
        0: 'b',
        1: 'm',
        2: 'y',
        3: 'g',
        4: 'r'
    }[class_id]


def plot_spike_average(elec):
    """
    This example plots last 100 spikes received by front end, calculate the
    average and plots it.
    """
    plt.figure(1)
    plt.xlabel('Time(ms)')
    plt.ylabel('uV')
    spk_count = 1
    spk_avg = empty([0, SPK_SAMPLES])
    spk_buffer = empty([SPK_BUFFER_SIZE, SPK_SAMPLES])
    spk_t = np.arange(0, 52000 / FS_CLK, 1000 / FS_CLK, dtype=np.float32)
    spk_buffer[:] = np.nan

    # Note: data starts collecting in a buffer *after* xipppy is instantiated.
    # Wait for 1 sec = 1000 samples/1000 Hz.
    # The data are concatenated into one array, the reason for time.sleep(1)
    # below.
    time.sleep(1)

    while plt.fignum_exists(1):
        _, seg_data = xp.spk_data(elec)
        for p in seg_data:
            if spk_count < SPK_BUFFER_SIZE:
                plt.title('Spike count: %s' % (spk_count))
                plt.plot(spk_t, p.wf, color=_unit_color(p.class_id),
                         linewidth=0.3)
                spk_buffer[spk_count] = p.wf
                spk_avg = np.nanmean(spk_buffer, axis=0)
                plt.pause(0.05)
                spk_count += 1
            else:
                plt.title('Electrode %s - Last %s spikes average'
                          % (elec, SPK_BUFFER_SIZE))
                plt.plot(spk_t, spk_avg, color='k', linewidth=4.0)
                plt.pause(0.1)
                plt.show()
                break


if __name__ == '__main__':
    with xp.xipppy_open():
        plot_spike_average(0)
