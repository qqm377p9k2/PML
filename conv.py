import __builtin__ as base
from numpy import *
import numpy.random as rand

import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import argparse
import sys
import time

import orthogonalMat, Walsh

def conv(a):
    n = a.shape[0]
    buffer = zeros(2**n)
    for i in xrange(n):
        buffer[2**i:2**(i+1)] = buffer[:2**i] + a[i]
    return buffer

def conv2d(a):
    n = a.shape[-1]
    buffer = zeros((2, 2**n))
    for i in xrange(n):
        buffer[:, 2**i:2**(i+1)] = buffer[:, :2**i] + 2*a[:, i][:,newaxis]
    return buffer - mean(buffer, axis=1)[:, newaxis]


def twoWalsh(n):
    b = ones((2,n))
    b[1,(n/2):] = -1.
    assert(abs(b[0].dot(b[1]))<1e-5)
    return b/n

def indep(n):
    b = zeros((2,n))
    b[0,:(n/2)] = 1
    b[1,(n/2):] = 1
    assert(abs(b[0].dot(b[1]))<1e-5)
    return b/(n/2.)

def hex(n):
    return asarray([[.5]*n +[.5]*n + [1.]*n,
                    [sqrt(3)/2]*n + [-sqrt(3)/2]*n + [0.]*n])

def dep(n, l):
    assert(n>l)
    b = zeros((2,n))
    b[0,:(n/2)] = 1
    b[0, (n/2):(n/2)+l] = 1
    b[1,(n/2):] = 1
    b[1, ((n/2)-l):(n/2)] = -1
    #assert(abs(b[0].dot(b[1]))<1e-5)
    return b/(n/2.+l)

def hist_(N=26, basis='orthogonalMat.sample'):
    import matplotlib
    b = eval(basis)(N)
    plt.hist2d(*conv2d(b[:2,:]), bins=160, norm=matplotlib.colors.LogNorm())
    plt.axis('equal')
    plt.show()

    
def middle(a):
    return asarray([a[:-1], a[1:]]).mean(axis=0)

def hist(lmd=[1.,1.], basis=orthogonalMat.sample(24), norm='L1', densitymap=False):
    b = basis
    if norm=='L1':
        norm = abs(b).sum(axis = 1)
        b = b/norm[:, newaxis]
        print norm
    hist,dx, dy = histogram2d(*conv2d(b[:2,:]), bins=160)
    extent = [dx.min(), dx.max(), dy.min(), dy.max()]
    print extent
    X,Y = meshgrid(middle(dx), middle(dy))
    loghist = log(hist) + lmd[0]*X**2 + lmd[1]*Y**2
    if not densitymap:
        plt.imshow(loghist, extent=extent, interpolation='nearest')
    else:
        plt.imshow(exp(loghist), extent=extent)
    plt.axis('equal')
    plt.colorbar()
    plt.show()

def H(p):
    return -p*log(p) - (1-p)*log(1-p) 


