"""
.. module:: ExSSA

ExSSA
*************

:Description: ExSSA

    

:Authors: bejar
    

:Version: 

:Created on: 29/06/2017 10:47 

"""
import numpy as np
from iWalker.Util.STFT import stft
import matplotlib.gridspec as gridspec
from iWalker.Util.Smoothing import ALS_smoothing, numpy_smoothing
from scipy.signal import argrelextrema
from iWalker.Data import User, Exercise, Exercises, Pacientes, Trajectory
from iWalker.Util import extract_extrema
from operator import itemgetter, attrgetter, methodcaller
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd
from sklearn.decomposition import PCA
import seaborn as sns
from iWalker.Util.SSA import SSA

__author__ = 'bejar'


if __name__ == '__main__':

    p = Pacientes()
    e = Exercises()
    p.from_db(pilot='NOGA')
    e.from_db(pilot='NOGA')
    for ex in e.iterator():
        t = Trajectory(ex.get_coordinates())
        if t.straightness()[0] < 0.95:
            e.delete_exercises([ex.id])

    e1 = e.iterator()

    ej = next(e1)

    L = 20
    mdata2 = np.array(ej.get_forces()[:, 1], dtype=float)
    # print (np.mean(mdata2))
    mdata2 -= np.mean(mdata2)
    ssa = SSA(n_components=L)

    ssa.fit(mdata2)

    fig = plt.figure(figsize=(60, 20))
    ax = fig.add_subplot(111)
    plt.plot(ssa.explained, 'r')

    plt.show()

    lseries = ssa.decomposition(range(L))

    # sseries = np.zeros(lseries[0].shape)
    # for i in range(L):
    #     fig = plt.figure(figsize=(60, 20))
    #     ax = fig.add_subplot(111)
    #
    #     sseries += lseries[i]
    #     plt.plot(lseries[i], 'r')
    #     plt.plot(mdata2, 'b')
    #     plt.show()
    aseries = np.array(lseries)
    corrmat = np.corrcoef(aseries)
    fig = plt.figure(figsize=(60, 20))
    ax = fig.add_subplot(111)
    sns.heatmap(corrmat, square=True)
    plt.show()

