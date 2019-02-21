"""
.. module:: Hilbert

Hilbert
*************

:Description: Hilbert

    

:Authors: bejar
    

:Version: 

:Created on: 07/04/2017 9:30 

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
from scipy.signal import hilbert, savgol_filter
from iWalker.Util.Smoothing import ALS_smoothing, numpy_smoothing

__author__ = 'bejar'


def plot_hilbert(ex):
    """
    Plots the exercise forces and their hilbert transform
    :param ex: 
    :return: 
    """
    f = 4
    forces = ex.get_forces()
    h_force = hilbert(forces[:,f])
    hi_force = np.imag(h_force)
    hr_force = np.real(h_force)
    e_force = np.abs(h_force)

    ip_force = np.unwrap(np.angle(h_force))
    if_force = np.diff(ip_force)/(2*np.pi)*10


    fig = plt.figure(figsize=(10, 16), dpi=100)
    axes = fig.add_subplot(4, 1, 1)
    axes.plot(range(forces.shape[0]), forces[:,f], 'r')
    axes.plot(range(forces.shape[0]), e_force, 'g')

    axes2 = fig.add_subplot(4, 1, 2)

    axes2.plot(range(forces.shape[0]), hr_force, 'r')
    axes2.plot(range(forces.shape[0]), hi_force, 'g')

    axes3 = fig.add_subplot(4, 1, 3)

    axes3.plot(range(forces.shape[0]-1), if_force, 'r')

    axes4 = fig.add_subplot(4, 1, 4)

    axes4.plot(np.real(h_force), np.imag(h_force), 'r')
    plt.show()


def plot_savgol(ex):
    """
    Plots the Savitzky-Golay, ALS_smoothing and other numpy filters applied to the forces
    :param ex: 
    :return: 
    """
    forces = ex.get_forces()
    f = 5

    fig = plt.figure(figsize=(10, 16), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    axes.plot(range(forces.shape[0]), forces[:,f], 'r')
    axes.plot(range(forces.shape[0]), savgol_filter(forces[:,f], 15, 3), 'g')
    axes.plot(range(forces.shape[0]), ALS_smoothing(forces[:,f], 1, 0.1), 'b')
    axes.plot(range(forces.shape[0]), numpy_smoothing(forces[:,f], 10, 'bartlett'), 'y')
    plt.show()


if __name__ == '__main__':

    # 'NOGA', 'FSL'
    p = Pacientes()
    e = Exercises()
    p.from_db(pilot='NOGA')
    e.from_db(pilot='NOGA')
    e.delete_patients(['FSL30'])

    for ex in e.iterator():
        # plot_hilbert(ex)
        plot_savgol(ex)
