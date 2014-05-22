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
import scipy.optimize as scopt

import argparse
import sys
import time
from basics import binary_expression, logSumExp

LBS = 15

MAX_N = 27

class Ising(object):
    def __init__(self, 
                 type = 'ST', 
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
            self.W = kwarg['half_bJ']*ones((self.N,self.N))/self.N
        elif type == 'HN':#Hopfield Net
            self.N = kwarg['N']
            ptns = (random.randn(kwarg['npatterns'], self.N) > 0)*2 - 1.
            self.ptns = ptns
            self.W = ptns.T.dot(ptns)/self.N
        elif type == 'RBM':
            N, M = kwarg['N'], kwarg['M']
            self.N = N+M
            self.W = zeros((N+M,N+M))
            self.W[:N, N:] = W
            self.W[N:, :N] = self.W[:N, N:].T
        self.eigen = safeEig(self.W)
        self.effective = abs(self.eigen[0]) > 1e-5

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
        
    def findroots(self, m0, debug=False):
        s, U = self.eigen
        effective = self.effective
        def fun(m):
            return m - U[:,effective].T.dot(tanh(2*sqrt(self.N)*m.dot((U*s)[:,effective].T)))/sqrt(self.N)
        def jac(m):
            sqsech = cosh(2*sqrt(self.N)*m.dot((U*s)[:,effective].T)).T**(-2.)
            return eye(effective.sum()) - 2.*(U[:,effective].T*sqsech).dot((U*s)[:,effective])
        if debug:
            return fun(m0), jac(m0)
        return asarray([scopt.fsolve(func=fun, fprime=jac, x0 = m0_) for m0_ in m0])

    def detjac(self, m, debug=False):
        '''
        Determinant of the complex free energy at a suddle point:
        \[
         f(\vec{m},\vec{\hat{M}}(m)) = - \sum_{k=1}^{n} \lambda_k m_k^2 - i\tr{\vec{m}}\vec{\hat{M}}(m) 
         - \frac{1}{N}\sum_{l=1}^{N} \log\left(2\cos\left(\sum_{j=1}^{n} u_{lj}\hat{M}_j(m) \right)\right)
        \]
        where the number of units is denoted by $N$, 
        the number of non-zero eigen values of the connection matrix $W$ is denoted by $n$, 
        $u_{ij}$ is the $i$th element of the $j$th eigen vector of $W$, 
        and $\vec{\hat{M}}(m)$ is defined as 
        \[
        \nabla_{\vec{m}} f(\vec{m},\vec{\hat{M}}(m)) = 0
        \]
        '''
        s, U = self.eigen
        effective = self.effective
        neff = effective.sum()
        def jac(m):
            sqsech = cosh(2*sqrt(self.N)*m.dot((U*s)[:,effective].T)).T**(-2.)
            return concatenate((concatenate((-2.*diag(s[effective]), -eye(neff)*1j),axis=1),
                                concatenate((-eye(neff)*1j, (U[:,effective].T*sqsech).dot(U[:,effective])), axis=1)))
        if debug:
            return linalg.det(jac(m)).real, jac(m)
        else:
            return linalg.det(jac(m)).real

    def free_energy(self, m):
        s, U = self.eigen
        eff = self.effective
        N = self.N
        return m.dot(m*s[eff]) - log(2.*cosh(2*sqrt(N)*m.dot((U*s)[:,eff].T))).sum(axis=0)/N

    @staticmethod
    def all_configs(N):
        bin_block = binary_expression(xrange(2**LBS), LBS)
        if N > LBS:
            for i in xrange(2**(N-LBS)):
                yield 2*concatenate((binary_expression(i*ones(2**LBS), N-LBS), bin_block), 1) -1.
        else:
            raise(NotImplementedError)

def test10(N=20, half_bJ=2.):
    ft = Ising(type='FT', N=N, half_bJ = half_bJ)
    m = arange(-2, 2, 0.02)
    obj, dobj = zip(*[ft.findroots(asarray([mm]), debug=True) for mm in m])
    obj = asarray([v[0] for v in obj])
    dobj= asarray([v[0] for v in dobj])
    plt.plot(m, obj)
    plt.plot(m, dobj)
    plt.show()
    return obj, dobj

def norm2(vec):
    return sqrt((vec*vec).sum(axis=len(vec.shape)-1))

def uniqueVec(a):
    '''heuristic routine for finding unique vectors'''
    a *= sign(a[:,0])[:, newaxis] ##remove the reflextional symmetry
    def foo(a,idx):
        a = a[argsort(a[:, idx])]
        ui = ones(len(a), 'bool')
        ui[1:] = abs(diff(a[:, idx], axis=0)) > 1e-5
        return a[ui]
    tmp = [foo(a, idx) for idx in random.permutation(a.shape[1])[:min(10, a.shape[1])]]
    assert(all(asarray([len(aa) for aa in tmp]) - len(tmp[0]) == 0))
    return tmp[0]

def test10_1(N=20, half_bJ=2.):
    ft = Ising(type='FT', N=N, half_bJ = half_bJ)
    m = random.rand(10)
    roots = ft.findroots(m[:, newaxis])
    roots = uniqueVec(roots)
    hessian = asarray([ft.detjac(m) for m in roots])
    fenergy = asarray([ft.free_energy(m) for m in roots])
    print hessian
    print roots
    argsup = hessian>0
    assert(any(argsup))
    logZ = log((exp(-ft.N*(fenergy[argsup]))/sqrt(hessian[argsup])).sum()) + log(2)
    print 'general routine:', logZ

    import AntiFerro_MF as MF
    MF.bJ = 2*half_bJ
    MF.N = N
    print 'FT specific1:', MF.computeZ2()
    print 'FT specific2:', MF.computeZ3()
    
    assert(N<27)
    print 'EXHAUSTIVE COMP.:', ft.computeZ()
    


def test11():
    net = Ising('HN', N=500, npatterns=100)
    roots = net.findroots(random.randn(200, 100))
    dist = asarray([sqrt(((roots-v)**2.).sum(axis=1)) for v in roots])
    plt.hist2d(*zip(*[[net.free_energy(m), sign(net.detjac(m))] for m in roots]), bins=20)
    plt.show()
    return roots, dist


def safeEig(symmat):
    s, U = linalg.eigh(symmat)
    UU = U.dot(U.T)
    assert(all((UU-ones(UU.shape))<1e-5))
    return s, U

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
