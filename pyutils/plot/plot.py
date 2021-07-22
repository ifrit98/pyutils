##############################################################################
# Plotting Routines                                                          #
##############################################################################


import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
matplotlib.use('agg')

#Defaults for legible figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams["image.cmap"] = 'jet'

# ALL_COLORS = list(colors.CSS4_COLORS)
COLORS = [
    'blue', # for original signal
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
] * 2


def histogram(x, bins='auto', show=True, save=False, outpath='histogram.png'):
    x = np.asarray(x).ravel()
    hist, bins = np.histogram(x, bins=bins)
    plt.bar(bins[:-1], hist, width=1)
    plt.savefig(outpath)
    if show:
        plt.show()

def plotx(y, x=None, xlab='obs', ylab='value', 
          title='', save=False, filepath='plot.png'):
    if x is None: x = np.linspace(0, len(y), len(y))
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel=xlab, ylabel=ylab,
        title=title)
    ax.grid()
    if save:
       fig.savefig(filepath)
    plt.show()

def poverlap(y, x=None):
    shp = np.asarray(y).shape
    if len(shp) < 2:
        raise ValueError("`y` must be at least 2D")
    if x is None: x = np.linspace(0, len(y[0]), len(y[0]))
    for i in range(shp[0]):
        plt.plot(x, y[i], COLORS[i])
    plt.show()

# 2 cols per row
def psubplot(y, x=None, figsize=[6.4, 4.8], filename=None, title=None):
    shp = np.asarray(y).shape
    if len(shp) < 2:
        raise ValueError("`y` must be at least 2D")
    if x is None: x = np.linspace(0, len(y[0]), len(y[0]))    
    i = 0
    _, ax = plt.subplots(nrows=shp[0]//2+1, ncols=2, figsize=figsize)
    for row in ax:
        for col in row:
            if i >= shp[0]: break
            col.plot(x, y[i], COLORS[i])
            i += 1
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

# one col per row
def psub1(y, x=None, figsize=[6.4, 4.8], filename=None, title=None, xlab=None, ylab=None, hspace=0.5):
    shp = np.asarray(y).shape
    if len(shp) < 2:
        raise ValueError("`y` must be at least 2D")
    if x is None: x = np.linspace(0, len(y[0]), len(y[0]))    
    i = 0
    fig, ax = plt.subplots(nrows=shp[0], ncols=1, figsize=figsize)
    for row in ax:
        if i >= shp[0]: break
        row.plot(x, y[i], COLORS[i])
        i += 1
    fig.subplots_adjust(hspace=hspace)
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    if title:
        ax[0].set_title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

psub = psubplot
polp = poverlap

def specgram(x, fs=1.0):
    from scipy import signal
    if not is_numpy(x):
        x = np.asarray(x)
    onesided = x.dtype.name not in ['complex64', 'complex128']
    f, t, Sxx = signal.spectrogram(x, fs, return_onesided=onesided)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.show()
