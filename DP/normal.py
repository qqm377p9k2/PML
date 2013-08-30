import numpy as np
import math
import matplotlib.pyplot as plt
import baseDist as bd


class normalDist(bd.baseDist):
    def __init__(self, mean=0, cov=1):
        self.mean = mean
        self.cov = cov

    def sample(self):
        pass

    def Zpost(self, observation):
        pass

    class likelihoodFun(bd.baseDist.likelihoodFun):
        def likelihood(self, data):
            pass
