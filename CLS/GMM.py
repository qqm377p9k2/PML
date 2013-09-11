import matplotlib.pyplot as plt
import __builtin__ as base
from numpy import *
from numpy.random import *

class GMM(object):
    def __init__(self, N):
        self.__data = []
        self.__dists = []
        self.__N = N
        self.__Ns = []
        self.__ratios = [1.]

    def append(self, dist, ratio=None):
        assert(isinstance(dist, natDist))
        if ratio is None:
            N = sum(array(self.__Ns))
            assert(N<self.__N)
            N = self.__N - N
        else:
            assert((ratio<1.) and  (ratio>0.))
            assert(ratio<self.__ratios[-1])
            self.__ratios.append(ratio)
            self.__ratios[-2] -= ratio
            N = self.__ratios[-2]*self.__N
        self.__Ns.append(N)
        self.__dists.append(dist)

    def sample(self):
        assert(sum(array(self.__ratios))==1.)
        assert(sum(array(self.__Ns))==self.__N)
        self.__data = [d.sample(N) for d,N in zip(self.__dists, self.__Ns)]
        return self
            
    def data(self):
        return self.__data

    def labels(self):
        return [ones(self.__Ns[i])*i for i in range(len(self.__dists))]

    def mixtures(self):
        t = concatenate(self.labels())
        x = concatenate(self.data())
        idcs = permutation(len(t))
        return (t[idcs],x[idcs,:])
    
class natDist(object):
    def __init__(self,mu, cov):
        dim = mu.shape[0]
        assert(mu.shape == (dim,))
        assert(cov.shape == (dim,dim))
        assert(all(cov.T == cov))
        self.__mu = mu
        self.__cov = cov
        self.__dim = dim

    def dim(self):
        return self.__dim

    def sample(self,N):
        (zvar, zrot) = linalg.eig(self.__cov)
        return dot(randn(N,self.__dim)*sqrt(zvar),zrot) + self.__mu


def main():
    gmm = GMM(N=1000)
    gmm.append(natDist(array([10.,10.]), 
                       array([[5.,0.],
                              [0.,3.]])),
               0.5)
    gmm.append(natDist(array([10.,-10.]),
                       array([[5.,0.],
                              [0.,5.]])))
    
    (t,x) = gmm.sample().mixtures()
    colors = [['blue', 'red'][int(label)] for label in t]
    plt.scatter(x[:,0], x[:,1], color=colors)
    dat = gmm.data();
    for i in range(len(dat)):
        nd = dat[i]-mean(dat[i],axis=0)
        print(dot(nd.T, nd)/nd.shape[0])
    plt.show()
    

if __name__=="__main__":
    main()
