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

def H(p):
    return -p*log(p) - (1-p)*log(1-p) 

J = 3.
J = -1.
N = 500

delta = 5e-3
fig = plt.figure()
#ax = fig.gca(projection='3d')
X = arange(delta, 1, delta)
Z = H(X) + J*(X-.5)**2
Z *= N
Z = exp(Z)

plt.plot(X, Z)

#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

