import numpy
import scipy
import matplotlib.pyplot
from plotly import express
from pandas import DataFrame
from scipy.signal import decimate, detrend


def icn_plot_raw_signals(times_or_fsample, raw_signals, channel_names=None,html_filename=False):
    
    """
    Creates a matplotlib figure with option to save as html file with plotly

    Arguments:
        time_or_fsample: input can be a times vector (in seconds) or the sampling rate (e.g. 250 Hz)
        raw_signals: a 2D-array of signals
        channels_names (optional): numpy array (or list) of channel names
        html_filename (optional): title and filename of the plot to save it as plotly html

    raw_signals = numpy.asarray(numpy.random.sample([5,1000]))
    times_or_fsample = 50

    returns nothing
    
    Example: 
        1) plot 20 seconds of random data with a putative sampling rate of 50 Hz
            icn_plot_raw_signals(50, numpy.random.sample(1000))
        2) plot 10 seconds of random data from 10 putative channels and a sampling rate of 1000 Hz
            icn_plot_raw_signals(1000, numpy.random.sample((10,20000)))
        3) add channel names to three raw signals sampled at 100 Hz
            icn_plot_raw_signals(100, numpy.random.sample((3,20000)),['ECoGL1','ECoGL2','STNL01'])
        4) save the last version as a plotly html file
            icn_plot_raw_signals(100, numpy.random.sample((3,20000)),['ECoGL1','ECoGL2','STNL01'],"Combined_LFP_ECoG_plot")
    """

    if len(raw_signals.shape) == 1:
        raw_signals = numpy.array(raw_signals,ndmin=2)
        
    if raw_signals.shape[1] > raw_signals.shape[0]:
        raw_signals = raw_signals.transpose()
   
    if isinstance(times_or_fsample,int) or len(times_or_fsample) == 1:
        times_or_fsample = numpy.linspace(0,raw_signals.shape[0]/times_or_fsample,raw_signals.shape[0])

    for i in range(0,raw_signals.shape[1]):
        raw_signals[:,i] = scipy.stats.zscore(raw_signals[:,i])/10+i
    
    if not channel_names:
        channel_names = list()
        for i in range(0,raw_signals.shape[1]):
            channel_names.append('channel_' + str(i))



    matplotlib.pyplot.plot(times_or_fsample, raw_signals)
    matplotlib.pyplot.yticks(range(0,raw_signals.shape[1]),channel_names,rotation=45)
    matplotlib.pyplot.xlim((times_or_fsample[0], times_or_fsample[-1]))
    matplotlib.pyplot.ylim((-1,raw_signals.shape[1]))
    matplotlib.pyplot.xlabel('Time [s]')
    matplotlib.pyplot.ylabel('Channels')
    
    if html_filename: 
        matplotlib.pyplot.title(html_filename)
        signals_df = DataFrame(raw_signals, times_or_fsample, columns=channel_names)

        fig = express.line(signals_df, x=signals_df.index, y=signals_df.columns,
                       line_shape="spline", render_mode="svg",
                       labels=dict(index="Time [s]",
                                   value="Z",
                                   variable="Channel"), title=html_filename)
        fig.update_layout(yaxis=dict(tickmode='array',
                                 tickvals=list(range(0,raw_signals.shape[1])),
                                 ticktext=channel_names))
        fig.write_html(str(html_filename + ".html"))
    



def rms(data, axis=-1):
    """
    returns the Root Mean Square (RMS) value of data along the given axis
    """
    assert axis < data.ndim, "No {} axis for data with {} dimension!".format(axis, data.ndim)
    if axis < 0:
        return numpy.sqrt(numpy.mean(numpy.square(data)))
    else:
        return numpy.sqrt(numpy.mean(numpy.square(data), axis=axis))

    
def sig_plot(time_array, signals_array, channels_array, samp_freq, plot_title=None,
             do_decimate=True, do_normalize=True, do_detrend="linear", padding=2):
    """
    Creates the signals as a matplotlib plot but does not show or save the plot!

    Arguments:
        time_array: numpy array of time stamps (seconds)
        signals_array: a 2D-array of signals with shape (#channels, #samples)
        channels_array: numpy array (or list) of channel names
        samp_freq: sampling frequency (Hz)
        plot_title: Plot title (default is None)
        do_decimate: down-sampling (decimating) the signal to 200Hz sampling rate
            (default and recommended value is True)
        do_normalize: dividing the signal by the root mean square value for normalization
            (default and recommended value is True)
        do_detrend: The type of detrending.
            If do_detrend == 'linear' (default), the result of a linear least-squares fit to data is subtracted from data.
            If do_detrend == 'constant', only the mean of data is subtracted.
            else, no detrending
        padding: multiplication factor for spacing between signals on the y-axis
            For highly variant data, use higher values. default is 2 
    
    returns nothing
    
    Example:
        ```python
        times = numpy.linspace(0, 10, 2000)
        channels = numpy.array(["a", "b", "c"])
        signals = numpy.random.normal(size=(3, 2000))
        fsample = 200
        matplotlib.pyplot.figure(figsize=(16, 9))
        sig_plot(times, signals, channels, fsample, plot_title="Detrended raw signals",
                 do_decimate=True, do_normalize=True, do_detrend="linear", padding=6)
        matplotlib.pyplot.savefig("SignalsPlot.svg")
        matplotlib.pyplot.show()
        ```
    """
    if do_decimate:
        decimate_factor = min(10, int(samp_freq / 200))
        signals_array = decimate(signals_array, decimate_factor)
        time_array = decimate(time_array, decimate_factor)
    if do_detrend == "linear" or do_detrend == "constant":
        signals_array = detrend(signals_array, axis= 1, type=do_detrend, overwrite_data=True)
    if do_normalize:
        eps_ = numpy.finfo(float).eps
        signals_array = signals_array / (rms(signals_array, axis=1).reshape(-1, 1) + eps_)

    offset_point = padding * numpy.std(signals_array, axis=1)

    matplotlib.pyplot.title(plot_title)
    ax = matplotlib.pyplot.gca()
    for i, ch_label in enumerate(channels_array):
        matplotlib.pyplot.plot(time_array, signals_array[i] + i*offset_point[i], label=ch_label)
    matplotlib.pyplot.yticks(numpy.linspace(0, i*offset_point[i], i+1, endpoint=True), channels_array)
    matplotlib.pyplot.xlabel("time (s)")
    matplotlib.pyplot.ylabel("channels")

    
def sig_plotly(time_array, signals_array, channels_array, samp_freq, file_name, plot_title=None,
               do_decimate=True, do_normalize=True, do_detrend="linear", padding=2):
    """
    Creates (exports) the signals as an HTML plotly plot

    Arguments:
        time_array: numpy array of time stamps (seconds)
        signals_array: a 2D-array of signals with shape (#channels, #samples)
        channels_array: numpy array (or list) of channel names
        samp_freq: sampling frequency (Hz)
        file_name: name (and directory) for the exported html file
        plot_title: Plot title (default is None)
        do_decimate: down-sampling (decimating) the signal to 200Hz sampling rate
            (default and recommended value is True)
        do_normalize: dividing the signal by the root mean square value for normalization
            (default and recommended value is True)
        do_detrend: The type of detrending.
            If do_detrend == 'linear' (default), the result of a linear least-squares fit to data is subtracted from data.
            If do_detrend == 'constant', only the mean of data is subtracted.
            else, no detrending
        padding: multiplication factor for spacing between signals on the y-axis
            For highly variant data, use higher values. default is 2 
    
    returns nothing
    """
    
    time_array = numpy.squeeze(time_array)
    signals_array = numpy.squeeze(signals_array)
    channels_array = numpy.squeeze(channels_array)
    if signals_array.ndim == 1:
        signals_array = signals_array.reshape(1, -1)
    
    assert signals_array.shape[0] == channels_array.shape[0], \
        "signals_array ! channels_array Dimension mismatch!"
    assert signals_array.shape[1] == time_array.shape[0], \
        "signals_array ! time_array Dimension mismatch!"

    if do_decimate:
        decimate_factor = min(10, int(samp_freq / 200))
        signals_array = decimate(signals_array, decimate_factor)
        time_array = decimate(time_array, decimate_factor)
    if do_detrend == "linear" or do_detrend == "constant":
        signals_array = detrend(signals_array, axis= 1, type=do_detrend, overwrite_data=True)
    if do_normalize:
        eps_ = numpy.finfo(float).eps
        signals_array = signals_array / (rms(signals_array, axis=1).reshape(-1, 1) + eps_)
    else:
        print("We strongly recommend using normalization (setting do_normalize to True)!")

    # # single offset
    offset_value = padding * rms(signals_array)
    signals_array = signals_array + offset_value * (numpy.arange(len(channels_array)).reshape(-1, 1))
    # # individual offset
    # rms_value = rms(signals_array, axis=1)
    # offset_value = numpy.zeros(len(rms_value)).reshape(-1, 1)
    # offset_value[1:, 0] = numpy.cumsum(rms_value)[1:]
    # signals_array = signals_array + (padding * offset_value)

    signals_df = DataFrame(data=signals_array.T, index=time_array, columns=channels_array)

    fig = express.line(signals_df, x=signals_df.index, y=signals_df.columns,
                       line_shape="spline", render_mode="svg",
                       labels=dict(index="Time (s)",
                                   value="(a.u.)",
                                   variable="Channel"), title=plot_title)
    fig.update_layout(yaxis=dict(tickmode='array',
                                 tickvals=offset_value * numpy.arange(len(channels_array)),
                                 ticktext=channels_array))

    fig.write_html(str(file_name) + ".html")


def raw_plotly(mne_raw, file_name, t_slice=(), plot_title=None,
               do_decimate=True, do_normalize=True, do_detrend="linear", padding=2):
    """
    Creates (exports) the (sliced) MNE raw signal as an HTML plotly plot

    Arguments:
        mne_raw: MNE raw object (output of mne.io.read_raw_...)
        file_name: name (and directory) for the exported html file
        t_slice: tuple of `start` and `end` slice (seconds)
            example: `t_slice = (1, 5)` returns the 1s-5s slice
        plot_title: Plot title (default is None)
        do_decimate: down-sampling (decimating) the signal to 200Hz sampling rate
            (default and recommended value is True)
        do_normalize: dividing the signal by the root mean square value for normalization
            (default and recommended value is True)
        do_detrend: The type of detrending.
            If do_detrend == 'linear' (default), the result of a linear least-squares fit to data is subtracted from data.
            If do_detrend == 'constant', only the mean of data is subtracted.
            else, no detrending
        padding: multiplication factor for spacing between signals on the y-axis
            For highly variant data, use higher values. default is 2 

    returns nothing
    """
    samp_freq = int(mne_raw.info["sfreq"])
    channels_array = numpy.array(mne_raw.info["ch_names"])
    if t_slice:
        signals_array, time_array = mne_raw[:, t_slice[0]*samp_freq:t_slice[1]*samp_freq]
    else:
        signals_array, time_array = mne_raw[:, :]

    sig_plotly(time_array, signals_array, channels_array, samp_freq, file_name, plot_title=plot_title,
               do_decimate=do_decimate, do_normalize=do_normalize, do_detrend=do_detrend, padding=padding)
