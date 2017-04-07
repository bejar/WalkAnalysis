"""
.. module:: Boss_spectral

Boss_spectral
*************

:Description: Boss_spectral

    

:Authors: bejar
    

:Version: 

:Created on: 03/03/2017 9:39 

"""

__author__ = 'bejar'


from iWalker.Data import User, Exercise, Exercises, Pacientes, Trajectory
from iWalker.Util.Misc import show_list_signals
from iWalker.Util import Boss, boss_distance, euclidean_distance, bin_hamming_distance, hamming_distance,\
    cosine_similarity
from sklearn.manifold import MDS, Isomap, TSNE, SpectralEmbedding
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import BayesianGaussianMixture as Dirichlet
import matplotlib.colors as colors

# colors = "rgbymcykrgbymcyk"

if __name__ == '__main__':
    p = Pacientes()
    e = Exercises()

    p.from_db(pilot='NOGALES')
    e.from_db(pilot='NOGALES')

    # e.delete_exercises([1425290750])

    wlen = 128
    voclen = 3
    ncoefs = 5

    nseries = 0
    lcl = []

    print(len(e.edict))
    for ex in e.iterator():
        t = Trajectory(ex.get_coordinates())
        if t.straightness()[0] < 0.95:
            e.delete_exercises([ex.id])
    print(len(e.edict))

    for ex in e.iterator():
        forces = ex.get_forces()
        if forces.shape[0] > wlen:
            nseries += 1
            if 'FSL' in ex.uid:
                lcl.append('r')
            else:
                lcl.append('g')
        else:
            e.delete_exercises([ex.id])
    print(len(e.edict))

    mdist = np.zeros((nseries, nseries))
    print(nseries)

    for f in range(6):
        dseries = {}
        for ex in e.iterator():
            forces = ex.get_forces()
            if forces.shape[0] > wlen:
                dseries[str(ex.uid) + '#' + str(ex.id)] = forces[:, f]

        boss = Boss(dseries, 10, butfirst=True)
        boss.discretization_intervals(ncoefs, wlen, voclen)
        boss.discretize()
        lcodes = list(boss.codes.keys())
        for i in range(len(lcodes)):
            for j in range(i+1, len(lcodes)):
                # mdist[i,j] += bin_hamming_distance(boss.codes[lcodes[i]], boss.codes[lcodes[j]])
                # mdist[i,j] += euclidean_distance(boss.codes[lcodes[i]], boss.codes[lcodes[j]])
                mdist[i,j] += cosine_similarity(boss.codes[lcodes[i]], boss.codes[lcodes[j]])
                mdist[j, i] = mdist[i,j]
                # mdist[i,j] += (boss_distance(boss.codes[v1], boss.codes[v2]) + boss_distance(boss.codes[v2], boss.codes[v1]))/2

    lexer = list(boss.codes.keys())

    mdist /= np.max(mdist)
    fdata = mdist
    imap = SpectralEmbedding(n_components=3, affinity='precomputed')
    fdata = imap.fit_transform(fdata)

    # fig = plt.figure(figsize=(10, 10))
    # # ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)
    #
    # # plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)
    # plt.scatter(fdata[:, 0], fdata[:, 1], c=lcl)
    #
    # plt.show()

    dp = Dirichlet(n_components=10)

    dp.fit(fdata)

    lab = dp.predict(fdata)
    print(np.unique(lab))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)


    plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100, c=lab/len(np.unique(lab)), cmap=plt.get_cmap('jet') )
    # plt.scatter(fdata[:, 0], fdata[:, 1], c=[colors[i] for i in lab])
    plt.colorbar()
    plt.show()

    # classes = {}
    # for i in np.unique(lab):
    #     classes[i] = []
    # for i, ex in  zip(lab, lexer):
    #     eid = ex.split('#')[1]
    #     classes[i].append((ex.split('#')[0], eid,  e.edict[int(eid)].lamb))
    #
    # for i in classes:
    #     print(sorted(classes[i]))

