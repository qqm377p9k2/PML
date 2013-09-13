import matplotlib.pyplot as plt
import __builtin__ as base
from GMM import *
from numpy import *
from numpy.random import randn, rand, permutation
from numpy import linalg as LA


class linear(object):
    def __int__(self):
        self.wml = None
        self.cPlane = None

    def fit(self, x, t, target=0):
        t = 2.*((t==target)-.5)
        x_ = ones((x.shape[0],x.shape[1]+1))
        x_[:,0:2] = x
        self.wml = dot(t, dot(x_,LA.inv(dot(x_.T,x_)).T))

    def cPlane(self, x):
        return -(self.wml[0]*x+self.wml[2])/self.wml[1]

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
    
    cls = linear()
    cls.fit(x,t,target=1)
    print(cls.wml)

    xlim = [min(x[:,0]), max(x[:,0])]
    tics = arange(xlim[0]-3,xlim[1]+3,0.01)
    plt.xlim(xlim[0]-3,xlim[1]+3)
    plt.plot(tics, cls.cPlane(tics),
             color='black')
    
    plt.show()


if __name__=="__main__":
    main()
