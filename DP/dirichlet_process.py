import numpy as np
import nltk
import dirichlet as diri
import matplotlib.pyplot as plt
from nltk.corpus import brown

class draw:
    """A draw from a diriclet process """
    cnts = []
    theta = []
    alpha = 0.01
    def train(self, data):
        assert(self.alpha>0)
    def posterior(self):
        self.theta


def main():
    diri.npSampleDirichlet(1,(3,10))

if __name__=="__main__":
    main()





