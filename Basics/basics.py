import __builtin__ as base
from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

import cPickle
import gzip

def ReL(x):
    ans = log(1+exp(x))
    ans[x>500] = x[x>500]
    return ans

def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))  #edit
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)

    return tuple(ans)

def sigmoid(x):
    return 1/(1+exp(-x))

def isVector(array, size=None):
    vectorp = array.ndim==1
    if size:
        vectorp = vectorp&(len(vector)==size)
    return vectorp

def isMatrix(array, size=None):
    matrixp = array.ndim==2
    if size:
        matrix = matrixp&(all(array.shape==size))
    return matrixp
       
def isSquareMatrix(array, size=None):
    squarep = array.ndim==2
    squarep = squarep&(reduce(equal, array.shape))
    if size:
        squarep = squarep&(array.shape[0]==size)
    return squarep
    

def isSymmetricMatrix(array, size=None):
    assert(isSquareMatrix(array, size))
    diag = range(array.shape[0])
    symmetricp = symmetricp&(all(array==array.T))
    symmetricp = symmetricp&(all(array[diag, diag] == 0))
    return symmetricp


def binary_expression(num, nDigits):
    return asarray([mod(asarray(num, dtype=int)/base, 2) for base in 2**arange(nDigits-1,-1,-1)]).T

def possible_configs(nDigits=3):
    return [binary_expression(x, nDigits) for x in xrange(2**nDigits)]
        

def logSumExp_(logZ):
    logZmax = max(logZ)
    return log(sum(exp(logZ - asarray(logZmax)))) + logZmax

def logSumExp(logZ, axis=None):
    if axis is None:
        axis = len(logZ.shape) -1 
    logZmax = logZ.max(axis=axis)
    shape = copy(logZ.shape)
    shape[axis] = 1
    return log(sum(exp(logZ - logZmax.reshape(shape)), axis=axis)) + logZmax

def logDiffExp(logZ, axis=None):
    if axis is None:
        axis = len(logZ.shape) -1 
    logZmax = logZ.max(axis=axis)
    shape = copy(logZ.shape)
    shape[axis] = 1
    return log(diff(exp(logZ - logZmax.reshape(shape)), axis=axis)) + logZmax

def main():
    print(binary_expression(3,5))
    print(possible_configs(3))


def approx_matrix_inv(A, V_0=None, niter=100):
    '''
    Approximate an inverse of a matrix A
    by using an iterative method that is
    invented by Li and Li (2010)
    '''
    if V_0 is None:
        V_0 = 0.001*random.randn(*A.shape)
    V = V_0.copy()
    I = eye(A.shape[0], dtype=float)
    for i in xrange(niter):
        AV = A.dot(V)
        #V = V.dot(3*I - AV.dot(3*I-AV))
        V = V.dot(7*I + AV.dot(-21*I + AV.dot(35*I+AV.dot(-35*I + AV.dot(21*I+AV.dot(-7*I+AV))))))
    return V
        

if __name__=='__main__':
    main()

