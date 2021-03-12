import numpy
from plotly import express
from pandas import DataFrame
from scipy.signal import decimate


def rms(data, axis=-1):
    """
    returns the Root Mean Square (RMS) value of data along the given axis
    """
    assert axis < data.ndim, "No {} axis for data with {} dimension!".format(axis, data.ndim)
    if axis < 0:
        return numpy.sqrt(numpy.mean(numpy.square(data)))
    else:
        return numpy.sqrt(numpy.mean(numpy.square(data), axis=axis))


def sig_plotly(time_array, signals_array, channels_array, samp_freq, file_name,
               do_decimate=True, do_normalize=True, do_demean=True):
    """
    Creates (exports) the signals as an HTML plotly plot
    Arguments:
        time_array: numpy array of time stamps (seconds)
        signals_array: a 2D-array of signals with shape (#channels, #samples)
        channels_array: numpy array (or list) of channel names
        samp_freq: sampling frequency (Hz)
        file_name: name (and directory) for the exported html file
        do_decimate: down-sampling (decimating) the signal to 200Hz sampling rate
            (default and recommended value is True)
        do_normalize: dividing the signal by the root mean square value for normalization
            (default and recommended value is True)
        do_demean: removing the mean of the signals (not necessary for filtered signals)
            (default and recommended value is True)
    
    returns nothing
    """
    
    assert signals_array.shape[0] != channels_array.squeeze().shape[0], \
            "signals_array ! channels_array Dimension mismatch!"
    assert signals_array.shape[1] != time_array.squeeze().shape[0], \
        "signals_array ! time_array Dimension mismatch!"
    
    if do_decimate:
        signals_array = decimate(signals_array, int(samp_freq / 200))
        time_array = decimate(time_array, int(samp_freq / 200))
    if do_demean:
        signals_array = signals_array - signals_array.mean(axis=1).reshape(-1, 1)
    if do_normalize:
        signals_array = signals_array / rms(signals_array, axis=1).reshape(-1, 1)
    
    offset_value = 2 * rms(signals_array)  # RMS value
    signals_array = signals_array + offset_value * (numpy.arange(len(channels_array)).reshape(-1, 1))
    
    signals_df = DataFrame(data=signals_array.T, index=time_array, columns=channels_array)
    fig = express.line(signals_df, x=signals_df.index, y=signals_df.columns,
            line_shape="spline", render_mode="svg",
            labels=dict(index="Time (s)",
                        value="(a.u.)",
                        variable="Channel"))
    fig.update_layout(yaxis = dict(tickmode = 'array',
                         tickvals = offset_value * numpy.arange(len(channels_array)),
                         ticktext = channels_array))
    
    fig.write_html(str(file_name) + ".html")
