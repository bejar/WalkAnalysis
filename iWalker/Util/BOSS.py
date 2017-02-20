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
        self.samplig = sampling
        self. coefs = []

    def compute(self, ncoef, wsize):
        """
        Computes the BOSS codes for the signals, the word length is 2*ncoefs (real and imaginary part)

        :param ncoef:
        :param wsize:
        :return:
        """

        for s in self.series:
            self.coefs.append(mft(s, self.sampling, ncoef, wsize))





