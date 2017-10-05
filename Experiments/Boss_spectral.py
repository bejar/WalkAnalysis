"""
.. module:: Boss_spectral

Boss_spectral
*************

:Description: Boss_spectral

    

:Authors: bejar
    

:Version: 

:Created on: 03/03/2017 9:39 

"""

from iWalker.Data import User, Exercise, Exercises, Pacientes, Trajectory
from iWalker.Util.Misc import show_list_signals
from iWalker.Util import Boss, boss_distance, euclidean_distance, bin_hamming_distance, hamming_distance, \
    cosine_similarity
from sklearn.manifold import MDS, Isomap, TSNE, SpectralEmbedding
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import BayesianGaussianMixture as Dirichlet
import matplotlib.colors as colors
from sklearn.metrics import silhouette_score
from collections import Counter

__author__ = 'bejar'

# colors = "rgbymcykrgbymcyk"

if __name__ == '__main__':
    p = Pacientes()
    e = Exercises()

    p.from_db(pilot='NOGALES')
    e.from_db(pilot='NOGALES')

    e.delete_exercises([1424971539, 1424968950])
    e.delete_exercises([1425290750, 1425376956, 1425486861, 1425484520])

    # e.delete_exercises([1425376956, 1425486861, 1425571115, 1425487748, 1425571166, 1425571388,
    #                     1424968992, 1424969366, 1424969549])

    wlen = 64
    voclen = 4
    ncoefs = 4

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
        if forces.shape[0] > (wlen*1.5):
            nseries += 1
            if 'FSL' in ex.uid:
                lcl.append('r')
            else:
                lcl.append('g')
        else:
            e.delete_exercises([ex.id])

    mdist = np.zeros((nseries, nseries))
    print(nseries)

    for f in [0, 1, 2, 3, 4, 5]:
        dseries = {}
        for ex in e.iterator():
            forces = ex.get_forces()
            trajec = Trajectory(np.array(ex.frame.loc[:, ['epx', 'epy']]), exer=ex.uid + ' ' + str(ex.id))
            beg, nd, _ = trajec.find_begginning_end()
            if forces.shape[0] > (wlen*1.5) and (nd - beg) > (wlen*1.5):
                dseries[str(ex.uid) + '#' + str(ex.id)] = forces[beg:nd, f]


        boss = Boss(dseries, 10, butfirst=True)
        boss.discretization_intervals(ncoefs, wlen, voclen)
        boss.discretize()
        lcodes = sorted(list(boss.codes.keys()))
        for i in range(len(lcodes)):
            for j in range(i + 1, len(lcodes)):
                # mdist[i,j] += bin_hamming_distance(boss.codes[lcodes[i]], boss.codes[lcodes[j]])
                # mdist[i,j] += euclidean_distance(boss.codes[lcodes[i]], boss.codes[lcodes[j]])
                # mdist[i,j] += (boss_distance(boss.codes[lcodes[i]], boss.codes[lcodes[j]]) + boss_distance(boss.codes[lcodes[j]], boss.codes[lcodes[i]]))/2
                mdist[i, j] += cosine_similarity(boss.codes[lcodes[i]], boss.codes[lcodes[j]])
                mdist[j, i] = mdist[i, j]

    lexer = sorted(list(boss.codes.keys()))
    print('NEX= ', len(lexer))

    mdist /= np.max(mdist)

    fdata = mdist
    # fdata = 1-mdist

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

    dp = Dirichlet(n_components=10, max_iter=200)

    dp.fit(fdata)

    lab = dp.predict(fdata)
    print(np.unique(lab))
    print (Counter(lab))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)
    plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100, c=lab / len(np.unique(lab)),
                cmap=plt.get_cmap('jet'))
    # plt.scatter(fdata[:, 0], fdata[:, 1], c=lab/len(np.unique(lab)), cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.show()

    print('Silhouette=', silhouette_score(fdata, dp.predict(fdata)))

    classes = {}
    for i in np.unique(lab):
        classes[i] = []

    for i, ex in zip(lab, lexer):
        eid = ex.split('#')[1]
        classes[i].append((ex.split('#')[0], eid,  e.edict[int(eid)].lamb, len(e.edict[int(eid)].frame)))

    lsets = []
    for i in classes:
        print(sorted(classes[i]))
        ejset = set()
        for ei in sorted(classes[i]):
            ejset.add(ei[0])
        lsets.append((ejset))

        # if len(classes[i]) < 20:
        #     for ei in sorted(classes[i]):
        #         print(ei[0], ei[2], ei[3])
        #         exi = e.edict[int(ei[1])]
        #         exi.show_exercise()

    for ejs in lsets:
        print(sorted(ejs))