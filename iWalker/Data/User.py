"""
.. module:: user

user
*************

:Description: user

    

:Authors: bejar
    

:Version: 

:Created on: 21/07/2016 13:33 

"""

__author__ = 'bejar'

def is_int(v):
    try:
        a = int(v)
        return True
    except ValueError:
        return False

def is_float(v):
    try:
        a = float(v)
        return True
    except ValueError:
        return False

class User():

    def __init__(self, nfile):
        f = open(nfile + '.usr', 'r')
        self.data = {}
        for line in f:
            vals = line.split(':')
            if is_int(vals[1].strip()):
                self.data[vals[0].strip()] = int(vals[1].strip())
            elif is_float(vals[1].strip()):
                self.data[vals[0].strip()] = float(vals[1].strip())
            else:
                self.data[vals[0].strip()] = vals[1].strip()
        f.close()

    def get_attr(self, attr):
        return self.data[attr]

    def exist_attr(self, attr):
        return attr in self.data