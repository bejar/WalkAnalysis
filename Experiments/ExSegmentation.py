"""
.. module:: ExSegmentation

ExSegmentation
*************

:Description: ExSegmentation

    

:Authors: bejar
    

:Version: 

:Created on: 24/04/2017 9:21 

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

__author__ = 'bejar'


def segment_signal(signal, sbegin, send, smoothed=True):
    """
    Return a vector with only the extrema
    :param signal:
    :return:
    """
    def find_next(init, sbegin, vector):
        for i in range(init, len(vector)):
            if vector[i]>= sbegin:
                return i

    if smoothed:
        smthsigx = ALS_smoothing(signal, 1, 0.1)
    else:
        smthsigx = signal

    smax = argrelextrema(smthsigx, np.greater_equal, order=3)[0]
    smin = argrelextrema(smthsigx, np.less_equal, order=3)[0]

    indmin = find_next(0, sbegin, smin)
    indmax = find_next(0, smin[indmin], smax)

    ltuples = []

    while indmin < len(smin) and smin[indmin] < send:
        ltuples.append((smin[indmin],0))
        indmin += 1

    while indmax < len(smax) and smax[indmax] < send:
        ltuples.append((smax[indmax],1))
        indmax += 1

    ltuples = sorted(ltuples,key=itemgetter(0))

    while len(ltuples) >0 and ltuples[-1][1] == 0:
        del ltuples[-1]


    if len(ltuples) > 0:

        ltuplesres = [ltuples[0]]
        for i in range(1,len(ltuples),1):
            if ltuplesres[-1][1] != ltuples[i][1]:
                ltuplesres.append(ltuples[i])

        ltuples = []
        for i in range(0,len(ltuplesres),2):
            ltuples.append((ltuplesres[i][0], ltuplesres[i+1][0]))
        return ltuples
    else:
        return None


def cover_distance(a,b):
    """
    If the interval intersects, returns the value of the intersection else 0
    :param a: 
    :param b: 
    :return: 
    """
    vi = 0
    vf = 0
    if ((a[0] <= b[0]) and a[1] >= b[0]) or \
       ((b[0] <= a[0]) and b[1] >= a[0]):  # there is an intersection
        if a[0] >= b[0]:
            vi = a[0]
        else:
            vi = b[0]
        if a[1] <= b[1]:
            vf = a[1]
        else:
            vf = b[1]

    return np.abs(vi - vf)

def match_segmentation(sX, sY, sZ):
    """
    Matches a set of intervals so they have maximal intersection 
    
    :param sX: 
    :param sY: 
    :param sZ: 
    :return: 
    """
    matchXY = {}
    for i in sX:
        matchXY[i] = (None, 0)
        for j in sY:
            cd = cover_distance(i,j)
            if matchXY[i][1] < cd:
                matchXY[i] = (j, cd)

    matchXZ = {}
    for i in sX:
        matchXZ[i] = (None, 0)
        for j in sZ:
            cd = cover_distance(i,j)
            if matchXZ[i][1] < cd:
                matchXZ[i] = (j, cd)

    matchYZ = {}
    for i in sY:
        matchYZ[i] = (None, 0)
        for j in sZ:
            cd = cover_distance(i,j)
            if matchYZ[i][1] < cd:
                matchYZ[i] = (j, cd)

    return matchXY, matchXZ, matchYZ




if __name__ == '__main__':

    # 'NOGA', 'FSL'
    p = Pacientes()
    e = Exercises()
    p.from_db(pilot='NOGA')
    e.from_db(pilot='NOGA')
    for ex in e.iterator():
        t = Trajectory(ex.get_coordinates())
        if t.straightness()[0] < 0.95:
            e.delete_exercises([ex.id])

    for ex in e.iterator():
        print (ex.uid + '-' + str(ex.id))

        trajec = Trajectory(np.array(ex.frame.loc[:, ['epx', 'epy']]), exer=ex.uid + ' ' + str(ex.id))
        beg, nd, vdis = trajec.find_begginning_end()
        print(beg, nd)

        ltuplesX = segment_signal(ex.frame['rhfx'] - ex.frame['lhfx'], beg, nd)
        ltuplesY = segment_signal(ex.frame['rhfy'] - ex.frame['lhfy'], beg, nd)
        ltuplesZ = segment_signal(ex.frame['rhfz'] - ex.frame['lhfz'], beg, nd)

        # print(ltuplesX)
        # print(ltuplesY)
        # print(ltuplesZ)

        matchXY, matchXZ, matchYZ = match_segmentation(ltuplesX, ltuplesY, ltuplesZ)

        fig = plt.figure(figsize=(60, 20))
        ax = fig.add_subplot(111)

        mxv = np.max(ex.frame['rhfx']-ex.frame['lhfx'])
        mnv = np.min(ex.frame['rhfx']-ex.frame['lhfx'])
        mdv = (mxv+mnv)/2
        # plt.plot(vdis, ex.frame['rhfx'], c='g')
        plt.plot(vdis, ex.frame['lhfz'], c='b')
        plt.plot(vdis, ex.frame['rhfz'], c='k')



        for i in ltuplesX:
            plt.plot([vdis[i[0]], vdis[i[1]]], [mnv, mnv], 'r')

        for i in ltuplesY:
            plt.plot([vdis[i[0]], vdis[i[1]]], [mdv, mdv], 'r')

        for i in ltuplesZ:
            plt.plot([vdis[i[0]], vdis[i[1]]], [mxv, mxv], 'r')

        for i in matchXY:
            if matchXY[i][0] is not None:
                plt.plot([vdis[i[0]], vdis[matchXY[i][0][0]]], [mnv, mdv], 'r')
        for i in matchXZ:
            if matchXZ[i][0] is not None:
                plt.plot([vdis[i[0]], vdis[matchXZ[i][0][0]]], [mnv, mxv], 'r')
        for i in matchYZ:
            if matchYZ[i][0] is not None:
                plt.plot([vdis[i[0]], vdis[matchYZ[i][0][0]]], [mdv, mxv], 'r')




        plt.show()
        plt.close()