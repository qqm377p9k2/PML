import __builtin__ as base
from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

import cPickle
import gzip

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
    return [mod(int(num)/base, 2) for base in 2**arange(nDigits-1,-1,-1)]

def possible_configs(nDigits=3):
    return [binary_expression(x, nDigits) for x in xrange(2**nDigits)]
        

def main():
    print(binary_expression(3,5))
    print(possible_configs(3))


def pickle(filename, object):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(object, f)

def unpickle(filename):
    with gzip.open(filename, 'rb') as f:
        return (cPickle.load(f))


if __name__=='__main__':
    main()
