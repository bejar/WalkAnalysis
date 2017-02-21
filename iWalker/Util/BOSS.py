"""
.. module:: BOSS

BOSS
*************

:Description: BOSS

    

:Authors: bejar
    

:Version: 

:Created on: 15/02/2017 13:48 

"""

from iWalker.Util import mft
import numpy as np
from kemlglearn.preprocessing import Discretizer
import seaborn as sn
from collections import Counter

__author__ = 'bejar'


class Boss():
    """
    Computes the BOSS words for a series
    """
    def __init__(self, lseries, sampling):
        """

        :param series:
        :param sampling:
        """
        self.series = lseries
        self.sampling = sampling
        self.coefs = []

    def discretization_intervals(self, ncoef, wsize, vsize):
        """
        Computes the BOSS discretization for the signals, the word length is 2*ncoefs (real and imaginary part) except
        is there are coefficients that are zero

        :param ncoef:
        :param wsize:
        :return:
        """

        for s in self.series:
            coefs = mft(s, self.sampling, ncoef, wsize)
            lcoefs = []
            for i in range(coefs.shape[1]):
                lcoefs.append(coefs[:,i].real)
                lcoefs.append(coefs[:,i].imag)

            self.coefs.append(np.stack(lcoefs, axis=-1))

        X = np.concatenate(self.coefs)

        self.disc = Discretizer(method='frequency', bins=vsize)
        self.disc.fit(X)

    def discretize(self):
        """
        Computes the words for each time series
        :param series:
        :return:
        """
        vocabulary = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        def word(vec):
            """

            :param v:
            :return:
            """
            w = ''
            for v in vec:
                w += vocabulary[int(v)]
            return w

        for c in self.coefs:
            sdisc = self.disc.transform(c, copy=True).real
            prevw = word(sdisc[0])
            lvoc = [prevw]
            for i in range(1,sdisc.shape[0]):
                nword = word(sdisc[i])
                if nword != prevw:
                    lvoc.append(nword)
            print(Counter(lvoc))


if __name__ == '__main__':
    from iWalker.Data import User, Exercise, Exercises, Pacientes, Trajectory
    from iWalker.Util.Misc import show_list_signals
    p = Pacientes()
    e = Exercises()
    p.from_db(pilot='NOGA')
    e.from_db(pilot='NOGA')
    e.delete_patients(['FSL30'])

    lseries = []

    i = 0
    for ex in e.iterator():
        if i < 3:
            lseries.append(ex.get_forces()[:, 0])
        else:
            break
        i += 1

    boss = Boss(lseries, 10)

    boss.discretization_intervals(2, 32, 3)
    boss.discretize()


