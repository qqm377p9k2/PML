import numpy as np
import math
import matplotlib.pyplot as plt
import baseDist as bd

vgamma = np.vectorize(math.gamma)

class dirichletDist(bd.baseDist):
    """A class for dirichlet distribution"""
    __params = []
    __pset = []

    def __init__(self, params):
        assert(isinstance(params, (list, tuple)))
        self.__params = np.array(params)
        self.__pset = set(params)

    def dim(self):
        """The dimension of the distribution"""
        return len(self.__params)

    def sample(self, size=1):
        """Draw samples"""
        assert(size>0)
        sample = np.zeros([self.dim(), size])
        for alpha in self.__pset:
            idcs = self.__params == alpha
            sample[np.ix_(idcs, range(size))] = np.random.gamma(alpha,1.0,
                                                                (np.sum(idcs),size))
        sample = sample/np.sum(sample, axis=0)
        if size==1:
            sample = self.likelihood(sample[:,0])
        return sample

    def pdf(self, mu):
        """Probability Density Function"""
        mu = np.array(mu)
        assert(len(mu)==self.dim())
        assert(sum(mu)-1<1e-10)
        pstr = np.product(mu**(self.__params-1))
        Z = np.product(vgamma(self.__params))/math.gamma(np.sum(self.__params))
        return pstr/Z

    def Zpost(self, observation):
        """
        Returns the normalizing constant of the posterior,
        \int p(observation|\mu)p(\mu) d\mu
        """
        assert(observation in range(self.dim()))
        return float(self.__params[observation])/np.sum(self.__params)

    class likelihood(bd.baseDist.likelihood):
        def __init__(self, mu):
            self.__mu = mu

        def dim(self):
            return len(self.__mu)

        def computeL(self, data):
            assert(data in range(self.dim))
            return self.__mu[data]

        def likelihoodFun(self):
            return lambda data: self.__mu[data]

def logGamma(x):
    """approximate log(gamma(x)) for large x by using Stirling's approximation """
    if x < 170:
        return math.log(math.gamma(x))
    else:
        return (x-1)*math.log(x-1) - (x-1) + 0.5*math.log(2*math.pi*(x-1))

def logBeta(alpha):
    return np.sum(np.log(vgamma(alpha))) - logGamma(np.sum(alpha))
      
def npLogPDFDirichlet(mu, params):
    mu = np.array(mu)
    params = np.array(params)
    assert(len(mu)==len(params))
    assert(sum(mu)-1<1e-10)
    logPstr = np.sum((params-1)*np.log(mu))
    logZ = np.sum(log(vgamma(params))) - logGamma(np.sum(params))
    return logPstr - logZ


def main():
    #
    print(logGamma(501)-math.log(math.factorial(500)))
    print(dirichletDist([1.]*15).Zpost(1))
    print(dirichletDist([10.]*15).Zpost(1))
    #
    noSamples = 500;
    plt.subplots_adjust(hspace=0.4)
    tran = np.array([np.array([1,-1,0])/np.sqrt(2),np.array([-1,-1,2])/np.sqrt(6)])
    #alpha = 0.1
    plt.subplot(231)
    dd = dirichletDist([0.1]*3)
    samples = np.dot(tran, dd.sample(noSamples))
    plt.scatter(samples[0,:], samples[1,:])
    #alpha = 1.0
    plt.subplot(232)
    dd = dirichletDist([1.]*3)
    samples = np.dot(tran, dd.sample(noSamples))
    plt.scatter(samples[0,:], samples[1,:])
    #alpha = 10.0
    plt.subplot(233)
    dd = dirichletDist([10.]*3)
    samples = np.dot(tran, dd.sample(noSamples))
    plt.scatter(samples[0,:], samples[1,:])
    #
    plt.subplot(234)
    dd = dirichletDist([2,1,1])
    samples = np.dot(tran, dd.sample(noSamples))
    plt.scatter(samples[0,:], samples[1,:])
    #
    plt.subplot(235)
    dd = dirichletDist([1,2,1])
    samples = np.dot(tran, dd.sample(noSamples))
    plt.scatter(samples[0,:], samples[1,:])
    #
    plt.subplot(236)
    dd = dirichletDist([1,1,2])
    samples = np.dot(tran, dd.sample(noSamples))
    plt.scatter(samples[0,:], samples[1,:])
    #
    plt.show()



if __name__=="__main__":
    main()





