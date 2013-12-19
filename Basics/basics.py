import __builtin__ as base
from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

def sigmoid(x):
    return 1/(1+exp(-x))
