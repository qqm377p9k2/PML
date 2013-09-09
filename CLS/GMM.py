import matplotlib.pyplot as plt
import __builtin__ as base
from numpy import *
from numpy.random import *

class GMM(object):
    def __init__(self, N):
        self.__data = []
        self.label= []
        self.dist = []
        self.__N = N
        self.ratios = [1.]

    def append(self, dist, ratio=None):
        assert(isinstance(dist, natDist2D))
        if ratio is None:
            N = sum(array([d.N for d in self.dist]))
            assert(N<self.__N)
            N = self.__N - N
        else:
            assert((ratio<1.) and  (ratio>0.))
            assert(ratio<self.ratios[-1])
            self.ratios.append(ratio)
            self.ratios[-2] -= ratio
            N = self.ratios[-2]*self.__N
        dist.N = N
        self.dist.append(dist)

    def sample(self):
        assert(sum(array(self.ratios))==1.)
        self.__data = [d.gen() for d in self.dist]
        return self
            
    def data(self):
        return self.__data

    def labels(self):
        return [ones(self.dist[i].N)*i for i in range(len(self.dist))]

    def mixtures(self):
        return (concatenate(self.labels()),concatenate(self.data()))
    
class natDist2D(object):
    def __init__(self,mu, cov, N=None):
        self.mu = mu
        self.cov = cov
        self.N = N
        
    def gen(self):
        assert(not(self.N is None))
        (zvar, zrot) = linalg.eig(self.cov)
        return dot(randn(self.N,2)*sqrt(zvar),zrot.T) + self.mu


def main():
    gmm = GMM(N=1000)
    gmm.append(natDist2D(array([10.,10.]), 
                         array([[3.,1.],
                                [1.,3.]])),
               0.5)
    gmm.append(natDist2D(array([10.,-10.]),
                         array([[5.,1.],
                                [1.,5.]])))
    gmm.sample()
    data = gmm.data()
    plt.scatter(data[0][:,0], data[0][:,1], color='blue')
    plt.scatter(data[1][:,0], data[1][:,1], color='red')
    (t,x) = gmm.mixtures()
    plt.show()
    

if __name__=="__main__":
    main()
