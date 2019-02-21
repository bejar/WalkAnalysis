"""
.. module:: BossSpectralExplore

BossSpectralExplore
*************

:Description: BossSpectralExplore

    

:Authors: bejar
    

:Version: 

:Created on: 27/04/2017 14:14 

"""


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
from sklearn.metrics import silhouette_score
from itertools import product

__author__ = 'bejar'

# colors = "rgbymcykrgbymcyk"

if __name__ == '__main__':
    p = Pacientes()
    e = Exercises()

    p.from_db(pilot='NOGALES')
    e.from_db(pilot='NOGALES')
    e.delete_exercises([1424971539, 1424968950])
    e.delete_exercises([1425290750, 1425376956, 1425486861, 1425484520])

    # e.delete_exercises([1425290750])

    for wlen in [128]:
        # wlen = 128
        print('*'*20)
        print('WLEN=', wlen)
        for ex in e.iterator():
            t = Trajectory(ex.get_coordinates())
            if t.straightness()[0] < 0.95:
                e.delete_exercises([ex.id])
        nseries = 0
        for ex in e.iterator():
            forces = ex.get_forces()
            if forces.shape[0] > wlen:
                nseries += 1
            else:
                e.delete_exercises([ex.id])

        for voclen, ncoefs in product([6],[3,4,5,6, 7, 8]):
            print('-'*20)
            print('VOCL=', voclen)
            print('NCOEFS=', ncoefs)

            mdist = np.zeros((nseries, nseries))

            for f in [0,1,2,3,4,5]:
                dseries = {}
                for ex in e.iterator():
                    forces = ex.get_forces()
                    trajec = Trajectory(np.array(ex.frame.loc[:, ['epx', 'epy']]), exer=ex.uid + ' ' + str(ex.id))
                    beg, nd, _ = trajec.find_begginning_end()
                    if forces.shape[0] > (wlen*1.5) and (nd - beg) > (wlen*1.5):
                        dseries[str(ex.uid) + '#' + str(ex.id)] = forces[beg:nd, f]

                    # if forces.shape[0] > wlen:
                    #     dseries[str(ex.uid) + '#' + str(ex.id)] = forces[:, f]

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
            for comp in range(2, 11):
                imap = SpectralEmbedding(n_components=comp, affinity='precomputed')
                fdatat = imap.fit_transform(fdata)

                dp = Dirichlet(n_components=20)

                dp.fit(fdatat)

                lab = dp.predict(fdatat)
                print('Comp=', comp, 'NCL=', len(np.unique(lab)), 'Silhouette=', silhouette_score(fdatat, lab))


