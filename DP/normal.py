import numpy as np
import math
import matplotlib.pyplot as plt
import baseDist as bd
import dirichlet_process as DP

class normalDist(bd.baseDist):
    def __init__(self, mean=0, cov=1):
        self.mean = mean
        self.cov = cov

    def samplePost(self, obs):
        return math.sqrt(self.cov)/math.sqrt(1+self.cov)*np.random.randn() + (self.mean + self.cov*obs)/(1+self.cov)

    def Zpost(self, obs):
        return math.exp(-0.5*((obs-self.mean)**2)/(1+self.cov))/math.sqrt(2*math.pi*(1+self.cov))

    class lFunSet(bd.baseDist.lFunSet):
        """set of likelihood functions"""
        __allocUnit = 1000
        def __init__(self, dist, size=1000):
            self.__centers = np.zeros(size)
            self.__counter= np.zeros(size)
            self.__pointer = 0

        def append(self, theta):
            if not(self.__pointer in range(self.__counter.shape[0])):
                self.__centers= np.concatenate([self.__centers,np.zeros(self.__allocUnit)])
                self.__counter= np.concatenate([self.__counter,np.zeros(self.__allocUnit)])
            self.__centers[self.__pointer] = theta
            self.__counter[self.__pointer] = 1.0
            self.__pointer += 1
           
        def compute(self, obs):
            return np.exp(-0.5*(obs-self.__centers[range(self.__pointer)])**2)/math.sqrt(2*math.pi)

        def theta(self, table):
            if table in range(self.__pointer):
                return self.__centers[table]
            else:
                return False

class gaussianMixture(object):
    __N = 1000
    def __init__(self):
        self.__mean = np.array([0., -8., 8.])
        self.__mu = np.array([2., 4., 6.])
        self.__mu /= np.sum(self.__mu)

    def dataGen(self):
        biases = np.concatenate([np.zeros(200), -8.*np.ones(400), 8.*np.ones(600)])
        np.random.shuffle(biases)
        return biases + np.random.randn(len(biases))

    def pdf(self):
        mean = self.__mean[:,None]
        return lambda x: np.dot(self.__mu,
                                np.exp(-0.5*(x-mean)**2)/np.sqrt(2.*math.pi))
        
def main():
    gm = gaussianMixture()
    data = gm.dataGen()
    h, b = np.histogram(data, bins=50)
    c = (b[:-1] + b[1:])/2
    draw = DP.DPdraw(alpha=1, baseDist=normalDist(cov=10))

    x = np.arange(-20,20,0.1)

    MAXITR = 100

    for i in range(MAXITR):
        print('Iteration '+repr(i))
        draw.CRP(data)
        print('noClusters '+repr(draw.noClusters()))
        popularTables = np.argsort(draw.prior())[:-10:-1]
        for table in popularTables:
            if draw.theta(table):
                print(repr(draw.prior()[table]) + ': ' + repr(draw.theta(table)))
        mass = draw.prior()
        y = np.sum(draw.likelihoodFunctions().compute(x[:,None])*mass[:-1],axis=1)
        plt.plot(x,y,color=(float(i)/MAXITR,0,1.-float(i)/MAXITR))

    plt.plot(x,gm.pdf()(x), color='green')
    plt.scatter(data, np.zeros(len(data)), color='green')

    plt.show()

if __name__=="__main__":
    main()
