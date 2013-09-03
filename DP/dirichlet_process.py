import numpy as np
import baseDist as bd
import matplotlib.pyplot as plt

class DPdraw(object):
    """A draw from a Dirichlet process"""

    def __init__(self, alpha=0.1, baseDist=None):
        self.alpha = alpha
        if baseDist != None:
            self.baseDist = baseDist
        else:
            assert(False)
        assert(isinstance(self.baseDist, bd.baseDist))
        self.__lfs = baseDist.lFunSet(baseDist) #likelihood functions

    def CRP(self, data):
        """Chinese Restaurant Process implementation"""
        assert(self.alpha>0)
        posteriorFun = self.posterior
        for i in range(len(data)):
            #print(repr(i)+'\t')
            post = posteriorFun(data[i])
            table = np.random.multinomial(1, post)
            if table[-1] == 1:       #a new table is organized and guide the customer
                self.__lfs.append(self.baseDist.samplePost(data[i]))
            else:                    #guide the customer to the prefered table
                self.__lfs.countUp(int(table.nonzero()[0]))

    def noClusters(self):
        return self.__lfs.length()

    def posterior(self, sample):
        """
        computes the posterior of DP
        sample: index of the observed word for topic learning
        """
        posterior=np.zeros(self.noClusters()+1)
        posterior[:-1] = self.__lfs.compute(sample) * self.__lfs.counter()
        posterior[-1] = self.alpha * self.baseDist.Zpost(sample)
        posterior = posterior/np.sum(posterior)
        assert(bd.ispv(posterior))
        return posterior

    def theta(self, table):
        return self.__lfs.theta(table)

    def prior(self):
        """
        computes the prior of DP
        """
        prior = np.zeros(self.noClusters()+1)
        prior[:-1] = self.__lfs.counter()
        prior[-1] = self.alpha
        return prior/np.sum(prior)

    def likelihoodFunctions(self):
        return self.__lfs

def main():
    pass

if __name__=="__main__":
    main()





