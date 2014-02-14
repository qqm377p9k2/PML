import __builtin__ as base
from numpy import *
import functools as ftk

import sys

import time

import cPickle, gzip

from Data import *
from rbm import RBM, GRBM
from variedParam import * 

from basics import *
from grbm_M2N2 import monitor, monitorInit
from tinyGRBM import tinyGRBM
import time

def play(filename):
    monitorInit()
    with gzip.open(filename, 'rb') as record:
        info = cPickle.load(record)
        data = cPickle.load(record)
        for i in xrange(info['nSamples']):
            rbm = cPickle.load(record)
            monitor(rbm, data)
            #time.sleep(0.01)

def load(filename):
    with gzip.open(filename, 'rb') as record:
        info = cPickle.load(record)
        data = cPickle.load(record)
        rbms = []
        for i in xrange(info['nSamples']):
            rbms.append(cPickle.load(record))
    return {'info':info, 'data':data, 'rbms':rbms}

if __name__=='__main__':
    play(sys.argv[1])
        
