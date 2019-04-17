import matplotlib.pyplot as plt
import pandas as pd

# TODO memory leaks


def getDataFrameFromText(path, index=0, delim=',', header_index=0):
    return pd.read_csv(path, index_col=index, sep=delim, header=header_index)


def plotDF(df, path):
    df.plot()
    plt.savefig(path)  # inserire path


def getMatrixFromDataFrame(df):
    return df.values.tolist()
