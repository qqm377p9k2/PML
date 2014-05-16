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

def H(p):
    return -p*np.log(p) - (1-p)*np.log(1-p) 

'''
Self Consistent Eq
\[
m = \frac{1}{N} \sum \tanh(\beta J_0 m)
f(m,\hat{m}) = -0.5 \beta J_0 m^2 -im\hat{m} - \log(\cosh(\hat{m}))
F(m) = -log(\int d\hat{m} exp(-Nf(m, \hat{m})))
\]
'''

bJ = 3.9
obj = lambda m:m-np.tanh(bJ*m)
N = 500

x0  = 2*(rand.rand(10)-.5)
m0 = asarray([scopt.newton(obj, x0_, maxiter=500) for x0_ in x0])
m0.sort()
unique = [x for i, x in enumerate(m0) if not i or abs(x - m0[i-1])>1e-5]
print unique


H = [[-bJ,0., 0., 1.],
     [0., bJ, 1.,0.],
     [0., 1.,1.,0.],
     [1., 0., 0., 0.]]
l,U =  np.linalg.eig(H)
print l
print U

#######################

def ddF(m):
    return - bJ - 1/((m+1)*(m-1))


print '#'*100

print [ddF(m) for m in unique]
