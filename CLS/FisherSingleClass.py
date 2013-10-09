import matplotlib.pyplot as plt
import __builtin__ as base
import multiClassLinear as MCL
from GMM import *
from linear import *
from numpy import *
from numpy.random import randn, rand, permutation
from numpy import linalg as LA


class FisherSingleClass (object):
    def __init__(self, dim):
        self.w = None

    def fit(self, x, t):
        pass


def main():
    gmm = MCL.composeGMM(type)
    

if __name__=="__main__":
    main()
