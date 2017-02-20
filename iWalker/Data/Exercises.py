"""
.. module:: Exercises

Exercises
*************

:Description: Exercises

    

:Authors: bejar
    

:Version: 

:Created on: 26/07/2016 11:48 

"""


from pymongo import MongoClient
from iWalker.Private.Connection import mongoserverlocal
from iWalker.Data import Exercise

__author__ = 'bejar'

class Exercises:

    header = 'lhfx,lhfy,lhfz,rhfx,rhfy,rhfz,lnf,rnf,acc,magn,gyro,hbl,hbr,epx,epy,epo,ls,rs'

    def __init__(self):
        pass

    def from_db(self, pilot=None, id=None):
        """
        Gets a set of exercises from the DB

        :param pilot:
        :param user:
        :return:
        """

        client = MongoClient(mongoserverlocal)
        db = client.IWalker
        col = db['Exercises']
        if pilot is None and id is None:
            c = col.find()
        elif pilot is not None:
            c = col.find({'User ID': {'$regex': pilot + '.*'}})
        else:
            c = col.find({'Unix Time': id})


        self.edict = {}

        for d in c:
            e = Exercise()
            e.from_data(d)
            self.edict[d['Unix Time']] = e


    def iterator(self):
        """
        Patients iterator

        :return:
        """
        for p in sorted(self.edict):
            yield self.edict[p]

    def delete_patients(self, lpat):
        """
        Deletes patients with a specific id from the stucture
        :param par: List of patients to delete
        :return:
        """
        lkeys = []
        for p in self.edict:
            if self.edict[p].uid in lpat:
                lkeys.append(p)
        for p in lkeys:
            del self.edict[p]

    def delete_exercises(self, lex):
        """
        Deletes patients with a specific id from the stucture
        :param par: List of patients to delete
        :return:
        """
        for p in lex:
            del self.edict[p]

if __name__ == '__main__':
    from iWalker.Data import Trajectory
    p = Exercises()

    p.from_db(pilot='FSL')

    for v in p.iterator():
        data = [v.get_forces()[:, 2]]
        # Trajectory(v.get_coordinates()).plot_trajectory(show=True)
        trj = Trajectory(v.get_coordinates()).plot_over_trajectory(data)


    print (len(p.edict))
    p.delete_patients(['FSL30'])
    print (len(p.edict))
