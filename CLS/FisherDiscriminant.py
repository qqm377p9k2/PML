import matplotlib.pyplot as plt
import __builtin__ as base
import multiClassLinear as MCL
from GMM import *
from linear import *
from numpy import *
from numpy.random import randn, rand, permutation
from numpy import linalg as LA


class Discriminator (object):
    def __init__(self, dim):
        self.w = None

    def fit(self, x, t):
        pass


class twoClsDiscriminator (object):
    def __init__(self):
        self.w = None

    def fit(self, x, t):
        assert(len(set(t))==2)
        x1 = x[t == 0,:]
        x2 = x[t == 1,:]
        m1 = mean(x1, axis=0)
        m2 = mean(x2, axis=0)
        Sw = dot((x1-m1).T, (x1-m1))/x1.shape[0] + dot((x2-m2).T, (x2-m2))/x2.shape[0]
        self.__clusterCenters = (m1,m2)
        self.centerDiff = m2 - m1
        self.w = dot(LA.inv(Sw), (m2-m1).T)
        #print(self.w)
        #print((m2-m1).T)

    def clusterCenters(self):
        c = self.__clusterCenters
        return ([c[0][0], c[1][0]], 
                [c[0][1], c[1][1]])

    def wPlane(self, x):
        w = self.w
        y = -(w[0]*x)/w[1]
        return (x,y)

    def centerDiffPlane(self, x):
        w = self.centerDiff
        y = -(w[0]*x)/w[1]
        return (x,y)



def testTwoClsDiscriminator():
    gmm = MCL.composeGMM(3)
    (t,x) = gmm.sample().mixtures()
    colors = [['blue', 'red'][int(label)] for label in t]
    plt.scatter(x[:,0], x[:,1], color=colors)
    disc = twoClsDiscriminator();
    disc.fit(x,t)

    xlim = [min(x[:,0]), max(x[:,0])]
    ylim = [min(x[:,1]), max(x[:,1])]
    tics = arange(xlim[0]-3,xlim[1]+3,0.01)
    plt.xlim(xlim[0]-3,xlim[1]+3)
    plt.ylim(ylim[0]-3,ylim[1]+3)

    plt.scatter(*disc.clusterCenters(), color='black')
    plt.plot(*disc.centerDiffPlane(tics), color='gray')
    plt.plot(*disc.wPlane(tics), color='orange')
    

    plt.show()
    

def main():
    testTwoClsDiscriminator()
    
if __name__=="__main__":
    main()
