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

fig = plt.figure()
ax = fig.gca
a = 0.2
delta = a*1e-2
X = arange(delta, a, delta)
plt.plot(X, H(X)+H(a-X)) 
#plt.ylim(.0, 0.8)


#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
#        linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

