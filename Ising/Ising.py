import __builtin__ as base
from numpy import *
import numpy.random as rand
import theano as th
import theano.tensor as T

import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from utils import pickle, unpickle

from nltk.corpus import treebank
from collections import Counter

import matplotlib.pyplot as plt

import argparse
import sys
import time
from basics import binary_expression, logSumExp

LBS = 15

MAX_N = 27

class Ising(object):
    def __init__(self, 
                 type = ('ST'), 
                 **kwarg):
        if type == 'ST':
            if 'N' in kwargs.keys():
                self.N = kwarg['N']
                self.W = rand.randn(self.N, self.N)
                self.W += self.W.T
            elif 'W' in kwargs.keys():
                W = kwarg['W']
                assert(all(W-W.T < 1e-6))
                self.W = W
                self.N = W.shape[0]
            elif 's' in kwargs.keys():
                s = kwarg['s']
                N = len(s)
                W = rand.randn(N, N)
                W += W.T
                ss, v = linalg.eig(W)
                self.W = dot(v, dot(diag(s), v.T))
                self.N = N
        elif type == 'FT':
            self.N = kwarg['N']
            self.W = kwarg['half_bJ']*ones((self.N,self.N))
        elif type == 'RBM':
            N, M = kwarg['N'], kwarg['M']
            self.N = N+M
            self.W = zeros((N+M,N+M))
            self.W[:N, N:] = W
            self.W[N:, :N] = self.W[:N, N:].T
        self.eigen = linalg.eig(self.W)

    def computeZ2(self):
        assert(self.N < 30)
        s,v = self.eigen
        Z = 0.
        for c in Ising.all_configs(self.N):
            Z += sum(exp(sum((dot(c, v)**2 * s), axis=1)))
        return Z
            
    def computeZ(self):
        assert(self.N < 30)
        s,v = self.eigen
        logZ = logSumExp([logSumExp(sum((dot(c, v)**2 * s), axis=1)) for c in Ising.all_configs(self.N)])
        return logZ

    def computeZ3(self):
        assert(self.N < 30)
        s,v = self.eigen
        logZ = logSumExp([logSumExp(sum((dot(c, v)**2 * s), axis=1)) for c in Ising.all_configs(self.N)])
        return logZ

    def entropy(self, axis=0, method='Sampling'):
        if method == 'Counting':
            assert(self.N < 30)
            s,v = self.eigen
            v = v[:,axis]
            m = [dot(c, v) for c in Ising.all_configs(self.N)]
            return asarray(m).reshape(2**self.N)
        elif method == 'Sampling':
            M = 1000000
            s,v = self.eigen
            v = v[:,axis]
            c = sign(random.randn(self.N, M))
            m = dot(v, c)
            return asarray(m).reshape(M)
            
    def lindeberg(self):
        'outdated'
        s,v = self.eigen
        sg = v**2
        L2 = sum(sg, axis=0)
        return (sg/L2).max(axis=0)

    def init_points(self, n_samples):
        s,u = self.eigen
        ranges = abs(u).sum(axis=0)/sqrt(self.N)
        return randn(n_samples, self.n)*ranges/2.
        
    def findroots(self, m0):
        s, U = self.eigen
        effective = abs(s) > 1e-5
        def fun(m):
            return m - tanh(2*sqrt(self.N)*((s*U)[effective]).T * m)
        def jac(m):
            pass

    @staticmethod
    def all_configs(N):
        bin_block = binary_expression(xrange(2**LBS), LBS)
        if N > LBS:
            for i in xrange(2**(N-LBS)):
                yield 2*concatenate((binary_expression(i*ones(2**LBS), N-LBS), bin_block), 1) -1.
        else:
            raise(NotImplementedError)



def Walsh(p):
    def fun(n, W=asarray([[1,1],[1,-1]])):
        if n==0:
            return W
        else:
            return fun(n-1, W=concatenate((concatenate((W,W)), concatenate((W,-W))), axis=1))
    return fun(p)

def Walsh2(n):
    if n==0:
        return asarray([[1,1],[1,-1]])
    else:
        H = fun(n-1)
        return concatenate((concatenate((H,H)), concatenate((H,-H))), axis=1)

#@profile   
def test0():
    [i for i in Ising.all_configs(22)]
    ising = Ising(5)
    s,v = ising.eigen
    print ising.W - dot(v, dot(diag(s), v.T))

def test1():
    ising = Ising(16)
    s = ising.eigen[0]
    print s.shape, 
    #s = range(22)
    s *= 2
    #s[2:] = 0.0
    print s
    W = Walsh(3)/4.
    print dot(W, W.T)
    W = dot(W, dot(diag(s), W.T))
    ising2 = Ising(W = W)
    ising = Ising(s=s)
    print ising.computeZ()
    print ising2.computeZ()
    print ising.eigen[0]
    print ising2.eigen[0]
    print ising2.eigen[1]
    print [net.lindeberg() for net in (ising, ising2)]
    m = ising.entropy()
    plt.hist(m, bins=100)
    plt.show()
    m = ising2.entropy()
    plt.hist(m, bins=100)
    plt.show()

def test2():
    k = 8

    W = Walsh(k-1)
    W = W/sqrt(2.**k)#/sqrt(2**k)
    print W
    print dot(W, W.T)

    ising = Ising(2**k)
    s = ising.eigen[0]
    print s.shape, 
    s *= 2
    #s[2:] = 0.0
    print s
    W = dot(W, dot(diag(s), W.T))
    ising2 = Ising(W = W)
    ising = Ising(s=s)
    print ising.eigen[0]
    print ising2.eigen[0]
    print ising2.eigen[1]
    print [net.lindeberg() for net in (ising, ising2)]
    m = ising.entropy()
    plt.hist(m, bins=100)
    plt.show()
    m = ising2.entropy()
    plt.hist(m, bins=100)
    plt.show()


if __name__=='__main__':
    test2()
