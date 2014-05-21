import __builtin__ as base
import numpy as np
from numpy import asarray, arange
import numpy.random as rand

import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import scipy.optimize as scopt

import argparse
import sys
import time

import AntiFerro_MF as MF
from AntiFerro_MF import bJ, N
from Ising import Ising, MAX_N

assert(N< MAX_N)

ising = Ising(W=0.5*bJ/N*np.ones((N,N)))
print ising.eigen[0]
print ising.computeZ()
#print np.log(ising.computeZ2())
print MF.computeZ2()
print MF.computeZ3()
