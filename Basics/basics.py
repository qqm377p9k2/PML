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

