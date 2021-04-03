import numpy as np
from matplotlib import pyplot as plt
import mne
import scipy.io as spio
import matplotlib.animation as animation
import seaborn as sn

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

data = loadmat("plt_features.mat")

data_here = np.concatenate((np.expand_dims(data["Y_TEST_CON"], 1),
                            np.expand_dims(data["ECOG_RIGHT_0_CON_predict"], 1),
                            data["FEATURES"][:,4,:]),
                            axis=1)


# inspired by https://matplotlib.org/stable/gallery/animation/unchained.html#sphx-glr-gallery-animation-unchained-py

# Create new Figure with black background
fig = plt.figure(figsize=(4, 4), facecolor='black', dpi=200)

# Add a subplot with no frame
ax = plt.subplot(frameon=True)

# x axis size
X_SIZE = 50
X = np.linspace(-2, 2, X_SIZE)

# Generate line plots
lines = []
hue_colors = sn.color_palette("viridis", data_here.shape[1])
for i in range(data_here.shape[1]):
    xscale = 5 - i / 10.
    # reduction of the X extents for linewidth (thicker strokes on bottom)
    lw = 3 - i / 100.0
    ax.fill_between(1 * X, 0, i + data_here[:X_SIZE,i],
                           color=hue_colors[data_here.shape[1]-i-1], lw=lw, alpha=1)

# Set y limit (or first line is cropped because of thickness)
ax.set_ylim(-1, data_here.shape[1])

# No x ticks
ax.set_xticks([])

# set y tick position
ticks_ = [1.5, 3.5, 5,5.5, 6, 6.5, 7,7.5, 8, 8.5]

# set y tick labels
plt.yticks(ticks_, ["MOV", "PREDICT", r'$\theta$',
                         r'$\alpha$', r'$low \beta$', r'$high \beta$', \
                         r'$all \beta$', r'$low \gamma$', r'$high \gamma$', r'$all \gamma$'])


# set title
plt.title("XGBOOST\n movement prediction")

# set all axes keywords to white
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.xaxis.label.set_color('white')
plt.style.use('dark_background')

def update(counter):
    plt.style.use('dark_background')

    # counter will index new data batch to plot
    counter += 1
    if counter > 2810:
        counter = 0
    # Update data
    ax.collections.clear()
    for i in np.flip(np.arange(0, data_here.shape[1], 1)):

        if i >=2: # to have small alpa difference between lowest two lines
            ax.fill_between(1 * X, 0, ticks_[i] + data_here[counter:counter+X_SIZE, i],
                           color=hue_colors[data_here.shape[1]-i-1], lw=lw, alpha=0.5)
        else:
            ax.fill_between(1 * X, 0, ticks_[i] + 2*data_here[counter:counter+X_SIZE, i],
                           color=hue_colors[data_here.shape[1]-i-1], lw=lw, alpha=0.8)

    # if tight_layout is not called labels are cut during saving
    plt.tight_layout()

# Construct the animation, using the update function as the animation director.
anim = animation.FuncAnimation(fig, update, interval=1, frames=800)


# ffmpeg this needs to be installed!
plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\ICN_admin\ffmpeg\bin\ffmpeg.exe"
FFwriter=animation.FFMpegWriter(fps=20, extra_args=['-vcodec', 'libx264'])
anim.save('plswork.mp4', writer=FFwriter)


plt.show()
