import matplotlib.pyplot as plt
import __builtin__ as base
from GMM import *
from linear import *
from numpy import *
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

class mLinear(object):
    def __init__(self, M):
        self.classifiers = []
        for i in range(M):
            self.classifiers.append(linear())

    def fit(self, x, t):
        for tgt in range(len(self.classifiers)):
            self.classifiers[tgt].fit(x,t,target=tgt)

    def wml(self):
        return array([cls.wml for cls in self.classifiers]).T

    def crossPoint(self):
        w = self.wml()
        tmp = ones(w.shape)
        b = zeros(len(self.classifiers))
        assert(len(self.classifiers)==3)
        tmp[0:2,:] = w[0:2,:]
        b[:] = -w[-1,:]
        print(w)
        print(tmp)
        print(b)
        p = dot(b,LA.inv(tmp))
        print(p)
        return (p[:-1], p[-1])

        
def main():
    gmm = GMM(N=1000)
    gmm.append(natDist(array([0.,10.]), 
                       array([[5.,0.],
                              [0.,3.]])),
               0.4)
    gmm.append(natDist(array([-5.,-10.]),
                       array([[5.,0.],
                              [0.,5.]])),
               0.3)
    gmm.append(natDist(array([15.,15.]),
                       array([[5.,0.],
                              [0.,5.]])))
    (t,x) = gmm.sample().mixtures()
    colors = [['blue', 'red', 'green'][int(label)] for label in t]
    plt.scatter(x[:,0], x[:,1], color=colors)
    
    cls = mLinear(3)
    cls.fit(x,t)

    xlim = [min(x[:,0]), max(x[:,0])]
    ylim = [min(x[:,1]), max(x[:,1])]
    tics = arange(xlim[0]-3,xlim[1]+3,0.01)
    plt.xlim(xlim[0]-3,xlim[1]+3)
    plt.ylim(ylim[0]-3,ylim[1]+3)
    plt.plot(tics, cls.classifiers[0].cPlane(tics),
             color='black')
    plt.plot(tics, cls.classifiers[1].cPlane(tics),
             color='black')
    plt.plot(tics, cls.classifiers[2].cPlane(tics),
             color='black')
    p, val = cls.crossPoint()
    plt.scatter(p[0], p[1], color='orange')
    plt.show()



if __name__=="__main__":
    main()
