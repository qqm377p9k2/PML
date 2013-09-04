import numpy as np
import math
import nltk
import dirichlet_process as DP
from nltk.corpus import brown
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
            sample = sample[:,0]
        return sample

    def samplePost(self, obs, size=1):
        """Draw samples"""
        assert(size>0)
        params = self.__params
        params[obs] += 1
        sample = np.zeros([self.dim(), size])
        for alpha in set(params):
            idcs = params == alpha
            sample[np.ix_(idcs, range(size))] = np.random.gamma(alpha,1.0,
                                                                (np.sum(idcs),size))
        sample = sample/np.sum(sample, axis=0)
        if size==1:
            sample = sample[:,0]
        return sample


    def pdf(self, mu):
        """Probability Density Function"""
        mu = np.array(mu)
        assert(len(mu)==self.dim())
        assert(bd.ispv(mu))
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

    class lFunSet(bd.baseDist.lFunSet):
        """set of likelihood functions"""
        __allocUnit = 1000
        def __init__(self, dist, size=1000):
            self.__record = np.zeros((dist.dim(), size))
            self.__counter= np.zeros(size)
            self.__pointer = 0

        def append(self, theta):
            assert(bd.ispv(theta))
            if not(self.__pointer in range(self.__record.shape[1])):
                self.__record = np.concatenate([self.__record, np.zeros((self.__record.shape[0], self.__allocUnit))], axis=1)
                self.__counter= np.concatenate([self.__counter,np.zeros(self.__allocUnit)])
            self.__record[:,self.__pointer] = theta
            self.__counter[self.__pointer] = 1.0
            self.__pointer += 1
           
        def compute(self, sample):
            assert(sample in range(self.__record.shape[0]))
            return self.__record[sample, range(self.__pointer)]

        def theta(self, table):
            if table in range(self.__pointer):
                return self.__record[:,table]
            else:
                return False

        def lFunTables(self):
            return self.__record[:,:self.__pointer]


class document:
    """
    extract words from Brown corpus to compose a vector expression like
    [0,1,3,1,2,0]
    """
    def __init__(self):
        #text = [w.lower() for w in brown.words(categories="news")]
        #text = [w.lower() for w in brown.words()]
        text = [w[0].lower() for w in brown.tagged_words(categories=["news", "science_fiction", "humor", "religion"],
                                                         simplify_tags=True) if (w[1]=='NP') | (w[1]=='N')]
        #text = [w[0].lower() for w in brown.tagged_words(simplify_tags=True) if (w[1]=='NP') | (w[1]=='N')]
        #text = text[np.random.permutation(range(len(text)))[:30000]]
        fdist = nltk.FreqDist(text)
        self.__fdist = fdist
        self.__words = fdist.samples()
        self.__data  = [self.__words.index(d) for d in [w for w in text if w in self.__words]]
    def words(self):
        return self.__words
    def data(self):
        return self.__data


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
    assert(bd.ispv(mu))
    logPstr = np.sum((params-1)*np.log(mu))
    logZ = np.sum(log(vgamma(params))) - logGamma(np.sum(params))
    return logPstr - logZ


def main0():
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


def main():
    doc = document()
    words = doc.words()
    print('datasz ' + repr(len(doc.data())))
    #counts = [np.sum(np.array(np.array(doc.data())==w)) for w in range(len(words))]
    #counts = [float(c)/max(counts) for c in counts]
    #draw = DPdraw(alpha=10, baseDist=diri.dirichletDist(counts))
    draw = DP.DPdraw(alpha=.1, baseDist=dirichletDist([.05]*len(words)))
    for i in range(5):
        print('Iteration '+repr(i))
        #print(repr(draw.prior()))
        draw.CRP(doc.data())
        print('noClusters '+repr(draw.noClusters()))
        popularTables = np.argsort(draw.prior())[:-70:-1]
        wList = [[words[widx] 
                  for widx in np.argsort(draw.theta(table))[:-7:-1]]
                 for table in popularTables 
                 if isinstance(draw.theta(table),np.ndarray)]
        popularity = [draw.prior()[table] for table in popularTables 
                      if isinstance(draw.theta(table),np.ndarray)]
        colLen = [max(len(c) for c in b) for b in zip(*wList)]
        for row,prb in zip(wList, popularity):
            print(repr(prb) + 
                  '\t: ' + 
                  ' '.join(s.ljust(l) for s,l in zip(row, colLen)))
    return draw


if __name__=="__main__":
    main()





