import __builtin__ as base
from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate

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

def main():
    print(binary_expression(3,5))
    print(possible_configs(3))


if __name__=='__main__':
    main()
