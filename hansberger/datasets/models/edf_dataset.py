import numpy as np
import pyedflib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import mpld3
from .dataset import Dataset, DatasetKindChoice


def stackplot(marray, seconds=None, start_time=None, ylabels=None):
    """
    will plot a stack of traces one above the other assuming
    marray.shape = numRows, numSamples
    """
    tarray = np.transpose(marray)
    stackplot_t(tarray, seconds=seconds, start_time=start_time, ylabels=ylabels)


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


class EDFDataset(Dataset):

    def save(self, *args, **kwargs):
        self.kind = DatasetKindChoice.EDF.value
        super().save(*args, **kwargs)
        self.rows = len(self.get_matrix_data())
        self.cols = len(self.get_matrix_data()[0])
        super().save(*args, **kwargs)

    def edf_to_matrix(self):
        data = self.dataframe
        height = data.signals_in_file
        ret = []
        for i in range(height):
            ret.append(data.readSignal(i).tolist())
        return ret

    def plot_EDF(self):
        data = self.dataframe
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
        stackplot(sigbufs_plot[:, :n_plot], ylabels=signal_labels)

    @property
    def dataframe(self):
        return pyedflib.EdfReader(self.source.path)

    @property
    def plot(self):
        plt.figure(figsize=(10, 5))
        self.plot_EDF()
        figure = plt.gcf()
        html_figure = mpld3.fig_to_html(figure, template_type='general')
        plt.close()
        return html_figure

    def get_matrix_data(self):
        return self.edf_to_matrix()
