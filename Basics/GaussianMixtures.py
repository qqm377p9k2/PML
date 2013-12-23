import matplotlib.pyplot as plt
import __builtin__ as base
from numpy import *
from numpy.random import *

class GaussianMixture(object):
    """Gaussian Mixture Generator"""
    def __init__(self, N):
        self.__data = []
        self.dists = []
        self.noDataPoints = N
        self.__noDataPoints = []
        self.ratios = [1.]

    def append(self, dist, ratio=None):
        assert(isinstance(dist, normalDist))
        if ratio is None:
            N = sum(array(self.__noDataPoints))
            assert(N<self.noDataPoints)
            N = self.noDataPoints - N
        else:
            assert((ratio<1.) and  (ratio>0.))
            rest =  self.ratios[-1]
            assert(ratio<rest)
            self.ratios.append(rest-ratio)
            self.ratios[-2] = ratio
            N = self.ratios[-2]*self.noDataPoints
            print self.ratios
        self.__noDataPoints.append(N)
        self.dists.append(dist)

    def sample(self):
        assert(sum(array(self.ratios))==1.)
        assert(sum(array(self.__noDataPoints))==self.noDataPoints)
        self.__data = [d.sample(N) for d,N in zip(self.dists, self.__noDataPoints)]
        return self
            
    def data(self):
        return [slot.copy() for slot in self.__data]

    def labels(self):
        return [ones(self.__noDataPoints[i])*i for i in range(len(self.dists))]

    def mixtures(self):
        t = concatenate(self.labels())
        x = concatenate(self.data())
        idcs = permutation(len(t))
        return (t[idcs],x[idcs,:])
    
class normalDist(object):
    def __init__(self,mu, cov):
        dim = mu.shape[0]
        assert(mu.shape == (dim,))
        assert(cov.shape == (dim,dim))
        assert(all((cov.T - cov)<1e-6))
        self.mu = mu
        self.cov = cov
        self.dim = dim

    def sample(self,N):
        (zvar, zrot) = linalg.eig(self.cov)
        return dot(randn(N,self.dim)*sqrt(zvar),zrot) + self.mu


def rotation2D(theta):
    return asarray([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def main():
    gmm = GaussianMixture(N=1000)
    corr = array([[6.,0.],
                  [0.,1.]])
    trans= array([2*sqrt(6),0.])
    gmm.append(normalDist(trans, corr), 0.5)
    rot = rotation2D(-pi/4)
    corr = dot(dot(rot, corr), rot.T)
    trans= dot(trans, rot)
    gmm.append(normalDist(trans, corr))


    
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
