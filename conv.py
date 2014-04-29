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

delta = 2e-2
fig = plt.figure()
ax = fig.gca(projection='3d')
X = arange(-1+delta, 1, delta)
Y = arange(-1+delta, 1, delta)
X, Y = meshgrid(X, Y)
Z = H((X+1)/2) + H(((Y/(1-abs(X)))+1)/2) #+ 0.2*X**2 + 0.7*Y**2
Z[abs(X)+ abs(Y) > 1] = 0

ZZ = H((Y+1)/2) + H(((X/(1-abs(Y)))+1)/2)
ZZ[abs(X)+ abs(Y) > 1] = 0
#Z = exp(10*Z)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

