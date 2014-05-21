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

def entropy(p):
    return -p*np.log(p) - (1-p)*np.log(1-p)

'''
Self Consistent Eq
\[
m = \frac{1}{N} \sum \tanh(\beta J_0 m)
f(m,\hat{m}) = -0.5 \beta J_0 m^2 -im\hat{m} - \log(\cosh(\hat{m}))
F(m) = -log(\int d\hat{m} exp(-Nf(m, \hat{m})))
\]
'''

bJ = 1.999
N = 26
N = 20

def solve():
    obj = lambda m:m-np.tanh(bJ*m)
    x0  = 2*(rand.rand(10)-.5)
    m0 = asarray([scopt.newton(obj, x0_, maxiter=500) for x0_ in x0])
    m0.sort()
    return [x for i, x in enumerate(m0) if not i or abs(x - m0[i-1])>1e-5]

######################

def foo(m):
    return [-.5*(bJ + 1./np.cosh(bJ*m)**2.) + sign*np.sqrt(.25*(bJ - 1./np.cosh(bJ*m)**2.)**2. + 1) for sign in [-1, 1]]
def foo1(m):
    return np.linalg.eig([[-bJ, 1],
                          [1,  -np.cosh(bJ*m)**(-2.)]])

def ddf2_old(m, bJ_=None):
    if bJ_ is None:
        bJ_ = bJ
    return np.linalg.eig([[-bJ_,0., 0., 1.],
                          [0., bJ_, 1., 0.],
                          [0., 1.,np.cos(0.)**(-2.),0.],
                          [1., 0., 0., -np.cosh(bJ_*m)**(-2.)]])

def ddf2(m, bJ_=None):
    if bJ_ is None:
        bJ_ = bJ
    return 1 - bJ_/np.cosh(bJ_*m)**2

def f2(m):
    return f3(m)

def computeZ2():
    m0 = asarray(solve())
    hessian = ddf2(m0)
    argsup = hessian>0
    assert(np.any(argsup))
    return np.log((np.exp(-N*(f2(m0[argsup])))/np.sqrt(hessian[argsup])).sum())


#######################
def f1(m):
    return -0.5*bJ*m**2 - entropy((m+1)/2)
def ddF(m):
    return - bJ - 1/((m+1)*(m-1))

def ddf3(m):
    if bJ > 0:
        return bJ*(1. - bJ/(np.cosh(bJ*m)**2))
    else:
        return -bJ*(1. - bJ/(np.cos(bJ*m)**2))

def f3(m):
    return .5*bJ*m**2 - np.log(2*np.cosh(bJ*m))

######################

def computeZ3():
    unique = solve()
    if bJ > 0:
        return np.log(np.sqrt(N*bJ/(2.0*np.pi))
                      *asarray([np.exp(-N*(f3(x0))) * np.sqrt(2.0*np.pi/(N*ddf3(x0))) for x0 in unique
                                if ddf3(x0) > 0]).sum())
    if bJ < 0:
        return np.log(np.sqrt(-N*bJ/(2.0*np.pi))
                      *asarray([np.exp(-N*(f3(x0))) * np.sqrt(2.0*np.pi/(N*ddf3(x0))) for x0 in unique
                                if ddf3(x0) > 0]).sum())

######################
if __name__=='__main__':
    unique = solve()
    print unique
    for x0 in unique:
        l,U =  np.linalg.eig(ddf2(x0))
        print l
        print U
    print '#'*100
    print [ddF(m) for m in unique]

    delta = 5e-3
    X = arange(-1.5+delta, 1.5, delta)

    f1p = plt.plot(X, np.exp(-N*f1(X)))

    f3p_approx = [plt.plot(X, np.exp(-N*(f3(x0) + 0.5*ddf3(x0)*(X-x0)**2))) for x0 in unique
                  if ddf3(x0) > 0]

    print [ddf3(x0) for x0 in unique]
    #f3p = plt.plot(X, np.exp(-N*f3(X)))
    #plt.legend([f3p, f3p_approx[0]], ['f3', 'f3_approx'])
    plt.show()


