"""
.. module:: Misc

Misc
*************

:Description: Misc

    

:Authors: bejar
    

:Version: 

:Created on: 20/02/2017 10:35 

"""


from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from iWalker.Util import ALS_smoothing
from scipy.signal import argrelextrema

__author__ = 'bejar'

def extract_extrema(signal, smoothed=True):
    """
    Return a vector with only the extrema
    :param signal:
    :return:
    """
    if smoothed:
        smthsigx = ALS_smoothing(signal, 1, 0.1)
    else:
        smthsigx = signal

    smax = argrelextrema(smthsigx, np.greater_equal, order=3)
    smin = argrelextrema(smthsigx, np.less_equal, order=3)
    vext = np.array([np.nan] * len(signal))
    vext[smax] = smthsigx[smax]
    vext[smin] = smthsigx[smin]
    return vext.copy()


def show_list_signals(signals, legend=[]):
    """
    Shows a list of signals in the same picture
    :param signal1:
    :param signal2:
    :return:
    """
    cols = ['r', 'g', 'b', 'k', 'y', 'c']
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(40)

    minaxis = np.min([np.min(s) for s in signals])
    maxaxis = np.max([np.max(s) for s in signals])
    num = len(signals[0])
    sp1 = fig.add_subplot(111)
    sp1.axis([0, num, minaxis, maxaxis])
    t = np.arange(0.0, num, 1)
    for i, s in enumerate(signals):
        sp1.plot(t, s, cols[i])
    plt.legend(legend)
    plt.show()