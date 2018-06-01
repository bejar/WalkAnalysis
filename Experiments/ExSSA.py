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
from sklearn.decomposition import FastICA

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

    R = 5
    for ej in e.iterator():

        L = 10
        mdata2 = np.array(ej.get_forces()[:, 4]-ej.get_forces()[:, 5], dtype=float)
        # print (np.mean(mdata2))
        mdata2 -= np.mean(mdata2)
        ssa = SSA(n_components=L)

        ssa.fit(mdata2)

        # fig = plt.figure(figsize=(60, 20))
        # ax = fig.add_subplot(111)
        # plt.plot(ssa.explained, 'r')
        #
        # plt.show()

        lseries = np.array(ssa.decomposition(range(L)))
        print(lseries.shape)

        # sseries = np.zeros(lseries[0].shape)
        fig = plt.figure(figsize=(40, 80))

        nrow = int(L / 2) + 1 if L % 2 == 0 else int(L / 2) + 2
        for i in range(0, L, 2):
            ax = fig.add_subplot(nrow, 2, i + 1)
            plt.title(str(i) + ' ' + str(ssa.explained[i]))
            plt.plot(lseries[i], 'r')
            if i + 1 < len(lseries):
                ax = fig.add_subplot(nrow, 2, i + 2)
                plt.title(str(i + 1) + ' - ' + str(ssa.explained[i + 1]))
                plt.plot(lseries[i + 1], 'r')

        ax = fig.add_subplot(nrow, 2, L + 1)
        plt.plot(mdata2, 'b')

        ax = fig.add_subplot(nrow, 2, L + 2)
        # plt.plot(ssa.reconstruct(R), 'b')
        # plt.plot(np.log(ssa.s), 'r')
        # plt.plot(ssa.explained, 'r')
        exp = 0
        rec = np.zeros(mdata2.shape[0])
        j = 0
        while exp < 0.9:
            rec += lseries[j]
            exp += ssa.explained[j]
            j += 1
        plt.plot(rec, 'r')
        # for i in range(L):
        #     rec += lseries[i]
        #     plt.plot(rec)
        #
        #     sseries += lseries[i]
        #     plt.plot(lseries[i], 'r')
        #     plt.plot(mdata2, 'b')
        plt.show()


        L=4
        fica = FastICA(n_components=L)

        lseries = fica.fit_transform(lseries.T)
        print(lseries.shape)
        fig = plt.figure(figsize=(40, 80))

        nrow = int(L / 2) + 1 if L % 2 == 0 else int(L / 2) + 2
        for i in range(0, L, 2):
            ax = fig.add_subplot(nrow, 2, i + 1)
            plt.title(str(i) + ' ' + str(ssa.explained[i]))
            plt.plot(lseries[:,i], 'r')
            if i + 1 < lseries.shape[1]:
                ax = fig.add_subplot(nrow, 2, i + 2)
                plt.title(str(i + 1) + ' - ' + str(ssa.explained[i + 1]))
                plt.plot(lseries[:,i + 1], 'r')

        ax = fig.add_subplot(nrow, 2, L + 1)
        plt.plot(mdata2, 'b')
        ax = fig.add_subplot(nrow, 2, L + 2)
        plt.plot(lseries[:,0] + lseries[:,1], 'r')

        plt.show()
        # aseries = np.array(lseries)
        # corrmat = np.corrcoef(aseries)
        # fig = plt.figure(figsize=(20, 20))
        # ax = fig.add_subplot(111)
        # sns.heatmap(corrmat, square=True)
        # plt.show()
