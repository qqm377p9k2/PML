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

N = 64;

def Haar(m):
    if m == 0:
        return ones(N)
    if m == 1:
        b = ones(N)
        b[N/2:] = -1
        return b
    else:
        b = Haar(1)
        for i in xrange(m-1):
            b = b.reshape(2,N/2).T.reshape(N/4,4).T.reshape(N)
        return b

def randPattern():
    A = ((rand.randn(N,N)>0)-.5)*2
    tmp = dot(A.T, A)
    u,s,v = linalg.svd(A)
    plt.plot(s)
    plt.show()
    plt.matshow(tmp)
    plt.show()


def Fourier(m):
    x = arange(N)
    return (1-mod(m,2))*cos(2*pi*(ceil(m/2.)*x)/N) + mod(m,2)*sin(2*pi*(ceil(m/2.)*x)/N)

def agg(M):
    tmp = zeros((N,N))
    for m in xrange(M):
        b = Haar(m)
        tmp += dot(b[:, newaxis], b[newaxis, :])
    plt.matshow(tmp)
    plt.show()
    
