import numpy as np
import nltk
from nltk.corpus import brown
import dirichlet as diri
import matplotlib.pyplot as plt

class DPdraw:
    """A draw from a Dirichlet process"""
    __counts = []
    __theta = []

    def __init__(self, alpha=0.1, noWords=None, baseDist=None):
        self.alpha = alpha
        if noWords != None:
            self.baseDist = diri.dirichletDist([1]*noWords)
        elif baseDist != None:
            self.baseDist = baseDist
        else:
            assert(False)
        assert(isinstance(self.baseDist, bd.baseDist))

    def CRP(self, data):
        """Chinese Restaurant Process implementation"""
        assert(self.alpha>0)
        for i in range(len(data)):
            #print(repr(i)+'\t')
            post = self.posterior(data[i])
            table = np.random.multinomial(1, post)
            if table[-1] == 1:       #a new table is organized and guide the customer
                self.__counts.append(1)
                self.__theta.append(self.baseDist.sample())
            else:                    #guide the customer to the prefered table
                self.__counts[int(table.nonzero()[0])] += 1

    def noClusters(self):
        assert(len(self.__counts) == len(self.__theta))
        return len(self.__counts)

    def posterior(self, sample):
        """
        computes the posterior of DP
        sample: index of the observed word
        """
        assert(sample in range(self.baseDist.dim()))
        posterior=np.zeros(self.noClusters()+1)
        posterior[:-1] = np.array([t[sample] for t in self.__theta])* np.array(self.__counts)
        posterior[-1] = self.alpha * self.baseDist.Zpost(sample)
        posterior = posterior/np.sum(posterior)
        assert(np.sum(posterior)-1<1e-10)
        return posterior

    def prior(self):
        """
        computes the prior of DP
        """
        prior = np.zeros(self.noClusters()+1)
        prior[:-1] = self.__counts
        prior[-1] = self.alpha
        return prior/np.sum(prior)

def extractData():
    """
    extract words from Brown corpus to compose a vector expression like
    [0,1,3,1,2,0]
    """
    news_text = brown.words(categories="news")
    fdist = nltk.FreqDist([w.lower() for w in news_text])
    words = fdist.samples()[10:6000:40]
    data = [w for w in news_text if w in words]
    return  [words.index(d) for d in data]


def main():
    draw = DPdraw(alpha=10, noWords=150)
    data = extractData()
    for i in range(100):
        print('Iteration '+repr(i))
        print('noClusters '+repr(draw.noClusters()))
        print(repr(draw.prior()))
        draw.CRP(data)
    return draw

if __name__=="__main__":
    main()





