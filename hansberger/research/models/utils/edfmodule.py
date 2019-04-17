import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pyedflib


def stackplot(marray, save_to, seconds=None, start_time=None, ylabels=None):
    """
    will plot a stack of traces one above the other assuming
    marray.shape = numRows, numSamples
    """
    tarray = np.transpose(marray)
    stackplot_t(tarray, seconds=seconds, start_time=start_time, ylabels=ylabels)
    plt.savefig(save_to)   # inserire path


def stackplot_t(tarray, seconds=None, start_time=None, ylabels=None):
    """
    will plot a stack of traces one above the other assuming
    tarray.shape =  numSamples, numRows
    """
    data = tarray
    numSamples, numRows = tarray.shape
    # data = np.random.randn(numSamples,numRows) # test data
    # data.shape = numSamples, numRows
    if seconds:
        t = seconds * np.arange(numSamples, dtype=float)/numSamples
    # import pdb
    # pdb.set_trace()
        if start_time:
            t = t+start_time
            xlm = (start_time, start_time+seconds)
        else:
            xlm = (0, seconds)

    else:
        t = np.arange(numSamples, dtype=float)
        xlm = (0, numSamples)

    ticklocs = []
    ax = plt.subplot(111)
    plt.xlim(*xlm)
    # xticks(np.linspace(xlm, 10))
    dmin = data.min()
    dmax = data.max()
    dr = (dmax - dmin)*0.7  # Crowd them a bit.
    y0 = dmin
    y1 = (numRows-1) * dr + dmax
    plt.ylim(y0, y1)

    segs = []
    for i in range(numRows):
        segs.append(np.hstack((t[:, np.newaxis], data[:, i, np.newaxis])))
        # print "segs[-1].shape:", segs[-1].shape
        ticklocs.append(i*dr)

    offsets = np.zeros((numRows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    lines = LineCollection(segs, offsets=offsets,
                           transOffset=None,
                           )

    ax.add_collection(lines)

    # set the yticks to use axes coords on the y axis
    ax.set_yticks(ticklocs)
    # ax.set_yticklabels(['PG3', 'PG5', 'PG7', 'PG9'])
    # if not plt.ylabels:
    plt.ylabels = ["%d" % ii for ii in range(numRows)]
    ax.set_yticklabels(ylabels)

    plt.xlabel('time (s)')


def readEDF(path):
    return pyedflib.EdfReader(path)


def edfToMatrix(data):
    height = data.signals_in_file
    ret = []
    for i in range(height):
        ret.append(data.readSignal(i).tolist())
    return ret


def plotEDF(data, save_to):
    n = data.signals_in_file
    signal_labels = data.getSignalLabels()
    n_min = data.getNSamples()[0]
    sigbufs = [np.zeros(data.getNSamples()[i]) for i in np.arange(n)]
    for i in np.arange(n):
        sigbufs[i] = data.readSignal(i)
        if n_min < len(sigbufs[i]):
            n_min = len(sigbufs[i])
    n_plot = np.min((n_min, 2000))
    sigbufs_plot = np.zeros((n, n_plot))
    for i in np.arange(n):
        sigbufs_plot[i, :] = sigbufs[i][:n_plot]

    stackplot(sigbufs_plot[:, :n_plot], save_to, ylabels=signal_labels)
