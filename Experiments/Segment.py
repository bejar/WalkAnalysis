"""
.. module:: Segment

Segment
*************

:Description: Segment

    

:Authors: bejar
    

:Version: 

:Created on: 15/09/2016 12:22 

"""



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
from pylab import *
import numpy as np
from iWalker.Util.STFT import stft
import matplotlib.gridspec as gridspec
from iWalker.Util.Smoothing import ALS_smoothing, numpy_smoothing
from scipy.signal import argrelextrema
from iWalker.Data import User, Exercise, Exercises, Pacientes, Trajectory
from operator import itemgetter, attrgetter, methodcaller
from iWalker.Util import extract_extrema
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
    print(ltuples)

    while len(ltuples) >0 and ltuples[-1][1] == 0:
        del ltuples[-1]
    print(ltuples)

    if len(ltuples) > 0:

        ltuplesres = [ltuples[0]]
        for i in range(1,len(ltuples),1):
            if ltuplesres[-1][1] != ltuples[i][1]:
                ltuplesres.append(ltuples[i])

        print(ltuplesres)
        ltuples = []
        for i in range(0,len(ltuplesres),2):
            ltuples.append((ltuplesres[i][0], ltuplesres[i+1][0]))
        return ltuples
    else:
        return None


def segmentation(frame, traj=False):
    """
    L-R XYZ forces smoothed with extrema points

    :return:
    """

    vextX = extract_extrema(frame['rhfx'] - frame['lhfx'])
    vextY = extract_extrema(frame['rhfy'] - frame['lhfy'])
    vextZ = extract_extrema(frame['rhfz'] - frame['lhfz'])

    smthsigz = ALS_smoothing(frame['rhfz'] - frame['lhfz'], 1, 0.1)

    # Look for the beggining of the exercise

    trajec = Trajectory(np.array(ex.frame.loc[:, ['epx', 'epy']]), exer=ex.uid + ' ' + str(ex.id))
    beg, nd, vdis = trajec.find_begginning_end()
    print(beg, nd)

    ltuples = segment_signal(frame['rhfz'] - frame['lhfz'], beg, nd)

    plotseg = True
    if plotseg :
        fig = plt.figure(figsize=(60, 20))
        ax = fig.add_subplot(111)
        plt.plot(vdis, smthsigz, c='r')
        plt.plot(vdis, vextX, c='g', marker='o')
        plt.plot(vdis, vextY, c='r', marker='o')
        plt.plot(vdis, vextZ, c='b', marker='o')
        plt.title(ex.uid + '/' + str(ex.id))

        ax.annotate('', xy=(vdis[beg], 1),
                    xycoords='data',
                    xytext=(-15, 25), textcoords='offset points',
                    arrowprops=dict(facecolor='green', shrink=0.05),
                    horizontalalignment='center', verticalalignment='bottom',
                    )
        ax.annotate('', xy=(vdis[nd], vextZ[nd]),
                    xycoords='data',
                    xytext=(-15, 25), textcoords='offset points',
                    arrowprops=dict(facecolor='yellow', shrink=0.05),
                    horizontalalignment='center', verticalalignment='bottom',
                    )


        if ltuples is not None:
            for i,j in ltuples:

                ax.annotate(str(i), xy=(vdis[i], vextZ[i]),
                            xycoords='data',
                            xytext=(-15, 25), textcoords='offset points',
                            arrowprops=dict(facecolor='black', shrink=0.03),
                            horizontalalignment='center', verticalalignment='bottom',
                            )
                ax.annotate(str(j), xy=(vdis[j], vextZ[j]),
                            xycoords='data',
                            xytext=(-15, 25), textcoords='offset points',
                            arrowprops=dict(facecolor='red', shrink=0.03),
                            horizontalalignment='center', verticalalignment='bottom',
                            )
            plt.show()
        else:
            plt.show()
        
        plt.close()

    if ltuples is not None:
        fig = plt.figure(figsize=(60, 20))
        ax = fig.add_subplot(111)
        plt.plot(vdis, frame['rhfz'], c='r')
        plt.plot(vdis, frame['lhfz'], c='b')
        plt.plot(vdis, frame['lhfx'], c='g')
        plt.plot(vdis, frame['rhfx'], c='y')

        ax.annotate('', xy=(vdis[beg], 1),
                    xycoords='data',
                    xytext=(-15, 25), textcoords='offset points',
                    arrowprops=dict(facecolor='green', shrink=0.05),
                    horizontalalignment='center', verticalalignment='bottom',
                    )
        ax.annotate('', xy=(vdis[nd], vextZ[nd]),
                    xycoords='data',
                    xytext=(-15, 25), textcoords='offset points',
                    arrowprops=dict(facecolor='yellow', shrink=0.05),
                    horizontalalignment='center', verticalalignment='bottom',
                    )


        for i,j in ltuples:

            ax.annotate(str(i), xy=(vdis[i], frame['rhfz'][i]),
                        xycoords='data',
                        xytext=(-15, 25), textcoords='offset points',
                        arrowprops=dict(facecolor='black', shrink=0.03),
                        horizontalalignment='center', verticalalignment='bottom',
                        )
            ax.annotate(str(j), xy=(vdis[j], frame['rhfz'][j]),
                        xycoords='data',
                        xytext=(-15, 25), textcoords='offset points',
                        arrowprops=dict(facecolor='red', shrink=0.03),
                        horizontalalignment='center', verticalalignment='bottom',
                        )
        plt.show()
        plt.close()



vars = ['lhfx','lhfy','lhfz','rhfx','rhfy','rhfz','lnf','rnf','acc','magn',
        'gyro','hbl','hbr','epx','epy','epo','ls','rs']
vars2 = ['lhfx','lhfy','lhfz','rhfx','rhfy','rhfz','lnf','rnf','tilt','roll',
        'hbl','hbr','epx','epy','epo','ls','rs']


freq = 10/2  # Half the sampling frequence

psfftl, psffto = 64, 48


print('Freq Resolution=',((freq*1.0)/psfftl))
# Ban frequencies below 0.25 Hz
ban = int(0.25 / ((freq*1.0)/psfftl))-1

if __name__ == '__main__':

    # 'NOGA', 'FSL'
    p = Pacientes()
    e = Exercises()
    p.from_db(pilot='NOGA')
    e.from_db(pilot='NOGA')
    e.delete_patients(['FSL30'])

    for ex in e.iterator():

        print (ex.uid+ '-' + str(ex.id))
        segmentation(ex.frame)
