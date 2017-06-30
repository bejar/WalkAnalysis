"""
.. module:: Exercise

Exercise
*************

:Description: Exercise

    

:Authors: bejar
    

:Version: 

:Created on: 26/07/2016 10:06 

"""

import pandas as pd
from pymongo import MongoClient
from iWalker.Private.Connection import mongoserverlocal
from io import StringIO
from iWalker.Data import User
import numpy as np

__author__ = 'bejar'


class Exercise:
    """
    Class representing the data from an exercise
    """

    uid = None
    id = None
    frame = None
    lamb = None
    etype = None
    speeds = None

    def __init__(self):
        pass

    def from_data(self, ex):
        self.uid = ex['User ID']
        self.id = ex['Unix Time']
        self.frame = pd.read_csv(StringIO(ex['csv']), sep=',')
        self.lamb = ex['i-Walker lambda_right']
        if 'Exercise' in ex:
            self.etype = ex['Exercise']

    def from_file(self, dfile):
        self.user = User(dfile)
        self.id = self.user.get_attr('Unix Time')
        self.uid = self.user.get_attr('User ID')
        self.lamb = self.user.get_attr('i-Walker lambda_right')
        if self.user.exist_attr('Exercise'):
            self.etype = self.user.get_attr('Exercise')

        self.frame = pd.read_csv(dfile + '.csv', sep=',')

    def from_db(self, id):
        client = MongoClient(mongoserverlocal)
        db = client.IWalker
        col = db['Exercises']
        c = col.find_one({'Unix Time': id})

        self.id = id
        self.frame = pd.read_csv(StringIO(c['csv']), sep=',')
        self.uid = c['User ID']
        self.lamb = c['i-Walker lambda_right']
        if 'Exercise' in c:
            self.etype = c['Exercise']

    def get_coordinates(self):
        """
        Coordinates of the exercise
        :return:
        """
        return np.column_stack((self.frame['epx'], self.frame['epy']
                                ))

    def get_forces(self):
        """
        Coordinates of the exercise
        :return:
        """
        return np.column_stack((self.frame['lhfx'], self.frame['lhfy'], self.frame['lhfz'],
                                self.frame['rhfx'], self.frame['rhfy'], self.frame['rhfz']
                                ))

    def classify(self, criteria):
        """
        Classifies an exercise following a criteria

        Speed - Uses the integral of the difference of right and left speed
                If integral is close to zero it is a straight line
                if (right-left) is positive then there is a left turn, else a right turn
        :param criteria:
        :return:
        """

        if criteria == 'speed':
            diff = np.array(self.frame['rs']) - np.array(self.frame['ls'])
            sdiff = np.sum(diff)
            if sdiff < -500:
                print('Der', sdiff)
            elif sdiff > 500:
                print('Izq', sdiff)
            else:
                print('Straight', sdiff)

    def compute_speed(self, mtime):
        """
        Compute the speed of the trajectory of the exercise using the (x,y) positions and the frequency

        :param freq:
        :return:
        """

        self.speeds = np.zeros(len(self.frame['epx']))
        for i in range(len(self.frame['epx']) - 1):
            self.speeds[i] = np.sqrt(((self.frame.loc[i, 'epx'] - self.frame.loc[i + 1, 'epx']) ** 2) +
                                     ((self.frame.loc[i, 'epx'] - self.frame.loc[i + 1, 'epx']) ** 2)) / mtime
        return self.speeds
