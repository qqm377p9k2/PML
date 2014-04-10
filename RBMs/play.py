import __builtin__ as base
from numpy import * #dot, arange, ix_, newaxis, asarray, array, transpose, reshape, zeros, ones, diag, meshgrid, tile, concatenate
import numpy as np
import numpy.random as rand

import functools as ftk

import sys

import time

import cPickle, gzip
import argparse

from Data import *
from rbm import RBM, GRBM
from variedParam import * 

from basics import *
from grbm_M2N2 import monitor, monitorInit
from tinyGRBM import tinyGRBM
import time

def play(rbms, data, idcs=None, key=False):
    monitorInit(False)
    prompt = '%d:' if key else '%d '
    if idcs is None:
        idcs = xrange(len(rbms))
    for i in idcs:
        sys.stdout.write(prompt%i)
        sys.stdout.flush()
        if key:
            sys.stdin.readline()
        monitor(rbms[i], data)
        time.sleep(0.005)

def load(filename, debug=False):
    with gzip.open(filename, 'rb') as record:
        info = cPickle.load(record)
        data = cPickle.load(record)
        rbms = []
        for i in xrange(info['nSamples']):
            if debug:
                print i,
            rbms.append(cPickle.load(record))
    if debug:
        print ''
    return info, data, rbms

def logspacing(beg=1., end=100., n_steps=100):
    beg = float(beg)
    end = float(end)
    base = (end/beg)**(1/float(n_steps-1))
    return base**arange(n_steps)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='reconstruction')
    ap.add_argument('--max', type=int, nargs=1, default=[None])
    ap.add_argument('filename', default=None)
    ap.add_argument('--stop', default=False, action='store_true')
    args = ap.parse_args()
    assert(args.filename is not None)
    if args.mode == 'reconstruction':
        info, data, rbms = load(args.filename)
        #idcs = xrange(info['nSamples'])
        if args.max[0] is None:
            args.max[0] = info['nSamples']
        idcs = asarray(np.round(logspacing(1, args.max[0], 20)), dtype=np.int32).tolist()
        play(rbms, data, idcs, args.stop)
    else:
        info, data, rbms = load(args.filename)
        print rbms[0].sparsity
        
