import numpy as np
import nltk
import dirichlet as diri
import baseDist as bd
import matplotlib.pyplot as plt

class DPdraw:
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
        posterior[:-1] = self.__lfs.lFunVals(sample) * self.__lfs.counter()
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


def main():
    doc = diri.document()
    words = doc.words()
    print('datasz ' + repr(len(doc.data())))
    #counts = [np.sum(np.array(np.array(doc.data())==w)) for w in range(len(words))]
    #counts = [float(c)/max(counts) for c in counts]
    #draw = DPdraw(alpha=10, baseDist=diri.dirichletDist(counts))
    draw = DPdraw(alpha=3, baseDist=diri.dirichletDist([1.]*len(words)))
    for i in range(100):
        print('Iteration '+repr(i))
        #print(repr(draw.prior()))
        draw.CRP(doc.data())
        print('noClusters '+repr(draw.noClusters()))
        popularTables = np.argsort(draw.prior())[:-50:-1]
        for table in popularTables:
            if isinstance(draw.theta(table),np.ndarray):
                freqwIdx = np.argsort(draw.theta(table))[:-7:-1]
                freqw = [words[widx] for widx in freqwIdx]
                print(repr(draw.prior()[table]) + '\t: ' + '\t\t'.join(freqw))

    return draw

if __name__=="__main__":
    main()





