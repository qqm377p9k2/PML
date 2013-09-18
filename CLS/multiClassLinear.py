import matplotlib.pyplot as plt
import __builtin__ as base
from GMM import *
from linear import *
from numpy import *
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

class multiClassLinear(object):
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
        self.p = dot(b,LA.inv(tmp))
        self.val = -self.p[-1]
        self.p = tuple(self.p[:-1])
        return (self.p[:], self.val)

    def border(self, x, target=(0,1)):
        wml = self.wml()
        w = wml[:,target[0]]-wml[:,target[1]]
        y = -(w[0]*x+w[2])/w[1]
        lim = dot(c_[x, y, ones(len(x))], wml[:,target[0]])>self.val
        return (x[lim],y[lim])

    def borders(self, x, target=0):
        wml = self.wml()
        wset = [wml[:,0]-wml[:,1], wml[:,0]-wml[:,2]]
        ys = [-(w[0]*x+w[2])/w[1] for w in wset]

        lim = dot(c_[x, ys[0], ones(len(x))], wml[:,0])>self.val
        return (x[lim],ys[0][lim])
    

def composeGMM(type=1):
    if type == 1:
        gmm = GMM(N=1000)
        gmm.append(normalDist(array([-10.,5.]), 
                              array([[5.,0.],
                                     [0.,3.]])),
                   0.4)
        gmm.append(normalDist(array([-5.,-10.]),
                              array([[5.,0.],
                                     [0.,5.]])),
                   0.3)
        gmm.append(normalDist(array([15.,15.]),
                              array([[5.,0.],
                                     [0.,5.]])))
    elif type == 2:
        gmm = GMM(N=1000)
        gmm.append(normalDist(array([0.,5.]), 
                              array([[5.,0.],
                                     [0.,3.]])),
                   0.3)
        gmm.append(normalDist(array([-2.,-10.]),
                              array([[5.,0.],
                                     [0.,5.]])),
                   0.3)
        gmm.append(normalDist(array([-5.,15.]),
                              array([[5.,0.],
                                     [0.,5.]])))
    return gmm

def test(type):
    gmm = composeGMM(type)
    (t,x) = gmm.sample().mixtures()
    colors = [['blue', 'red', 'green'][int(label)] for label in t]
    plt.scatter(x[:,0], x[:,1], color=colors)
    
    cls = multiClassLinear(3)
    cls.fit(x,t)

    xlim = [min(x[:,0]), max(x[:,0])]
    ylim = [min(x[:,1]), max(x[:,1])]
    tics = arange(xlim[0]-3,xlim[1]+3,0.01)
    plt.xlim(xlim[0]-3,xlim[1]+3)
    plt.ylim(ylim[0]-3,ylim[1]+3)
    for c in range(3):
        plt.plot(*cls.classifiers[c].cPlane(tics), color='gray')
    p, val = cls.crossPoint()
    plt.scatter(*p, color='black')
    plt.plot(*cls.border(tics, target=(0,1)), color='black')
    plt.plot(*cls.border(tics, target=(1,2)), color='black')
    plt.plot(*cls.border(tics, target=(2,0)), color='black')
    plt.show()

def main():
    test(type=2)


if __name__=="__main__":
    main()
