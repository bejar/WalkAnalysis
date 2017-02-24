"""
.. module:: Exp_BOSS

Exp_BOSS
*************

:Description: Exp_BOSS

    

:Authors: bejar
    

:Version: 

:Created on: 22/02/2017 11:20 

"""

__author__ = 'bejar'

from iWalker.Data import User, Exercise, Exercises, Pacientes, Trajectory
from iWalker.Util.Misc import show_list_signals
from iWalker.Util import Boss, boss_distance, euclidean_distance
from sklearn.manifold import MDS, Isomap, TSNE, SpectralEmbedding
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    p = Pacientes()
    e = Exercises()
    p2 = Pacientes()
    e2 = Exercises()

    p.from_db(pilot='NOGALES')
    e.from_db(pilot='NOGALES')
    p2.from_db(pilot='FSL')
    e2.from_db(pilot='FSL')
    e2.delete_patients(['FSL30'])

    e.merge(e2)
    # e.delete_exercises([1424947438, 1424947235, 1425295876, 1425290750, 1425571115, 1424969280, 1424969226])
    # e.delete_exercises([1416241920, 1416241871, 1416409354, 1416391685, 1416933676, 1416918342, 1416391884, 1416391948])
    wlen = 32
    voclen = 3
    ncoefs = 3

    nseries = 0
    lcl = []
    for ex in e.iterator():
        forces = ex.get_forces()
        if forces.shape[0] > wlen:
            nseries += 1
            if 'FSL' in ex.uid:
                lcl.append('r')
            else:
                lcl.append('g')

    mdist = np.zeros((nseries, nseries))
    print(nseries)

    for i in range(6):
        dseries = {}
        for ex in e.iterator():
            forces = ex.get_forces()
            if forces.shape[0] > wlen:
                dseries[str(ex.uid) + '#' + str(ex.id)] = forces[:, i]

        boss = Boss(dseries, 10, butfirst=True)
        boss.discretization_intervals(ncoefs, wlen, voclen)
        boss.discretize()
        lcodes = list(boss.codes.keys())

        for i in range(len(lcodes)):
            for j in range(i+1, len(lcodes)):
                mdist[i,j] += euclidean_distance(boss.codes[lcodes[i]], boss.codes[lcodes[j]])
                mdist[j, i] = mdist[i,j]
                # mdist[i,j] += (boss_distance(boss.codes[v1], boss.codes[v2]) + boss_distance(boss.codes[v2], boss.codes[v1]))/2

    # lej = []
    # for i, ex in enumerate(boss.codes.keys()):
    #     lej.append((np.mean(mdist[i,:]), ex))
    #
    # for d, e in sorted(lej):
    #     print(e,d)

    mdist /= np.max(mdist)
    transf = MDS(n_components=50, dissimilarity='precomputed', n_jobs=-1, random_state=0)
    fdata = transf.fit_transform(mdist)
    print(transf.stress_)

    imap = Isomap(n_components=2, n_neighbors=7, n_jobs=-1)
    fdata = imap.fit_transform(fdata)


    fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    # plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)
    plt.scatter(fdata[:, 0], fdata[:, 1], c=lcl)

    plt.show()
