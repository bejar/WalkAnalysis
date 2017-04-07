"""
.. module:: Pacientes

Pacientes
*************

:Description: Pacientes

    

:Authors: bejar
    

:Version: 

:Created on: 26/07/2016 8:54 

"""

__author__ = 'bejar'


import pandas as pd
from pymongo import MongoClient
from iWalker.Private.Connection import mongoserver, mongouser, mongopass


class Pacientes():
    """
    Class for the patients data
    """

    def __init__(self):
        pass

    def from_file(self, dfile):
        """
        Load patients from csv file

        :param dfile:
        :return:
        """
        frame = pd.read_csv(dfile+'.csv', sep=';')
        self.ddict = {}

        for d in frame.itertuples():
            self.ddict[d.Codigo] = {'Codigo': d.Codigo, 'Edad': int(d.Edad), 'Tineti': int(d.Tineti),
                                    'Barthel': int(d.Barthel), 'GDS': int(d.GDS), 'Caidas': int(d.Caidas)}

    def from_db(self, pilot=None):
        """
        Load patients from mongo DB

        :return:
        """
        client = MongoClient(mongoserver)
        db = client.iwalkersws
        db.authenticate(mongouser, password=mongopass)
        col = db['Users']

        if pilot is None:
            c = col.find()
        else:
            c = col.find({'Codigo': {'$regex': pilot + '.*'}})


        self.ddict = {}
        for d in c:
            self.ddict[d['Codigo']] = {'Codigo': d['Codigo'], 'Edad':d['Edad'], 'Tineti': d['Tineti'], 'Barthel': d['Barthel'], 'GDS': d['GDS'], 'Caidas': d['Caidas']}


    def get_patient_attribute(self, patient, attribute):
        """
        Returns the data for a specific patient

        :param patient:
        :param attribute:
        :return:
        """
        if patient in self.ddict:
            return self.ddict[patient][attribute]
        else:
            return None

    def iterator(self):
        """
        Patients iterator

        :return:
        """
        for p in self.ddict:
            yield self.ddict[p]

    def del_patients(self, lpat):
        """
        Deletes patients with a specific id from the stucture
        :param par: List of patients to delete
        :return:
        """
        for p in lpat:
            del self.ddict[p]

if __name__ == '__main__':
    p = Pacientes()

    p.from_db(pilot='FSL')

    for v in p.iterator():
        print(v)
    print(len(p.ddict))
    p.del_patients(['FSL30'])


    print(len(p.ddict))