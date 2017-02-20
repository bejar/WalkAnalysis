"""
.. module:: Misc

Misc
*************

:Description: Misc

    

:Authors: bejar
    

:Version: 

:Created on: 20/02/2017 10:35 

"""

__author__ = 'bejar'

from pylab import *
import matplotlib.pyplot as plt
import numpy as np

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