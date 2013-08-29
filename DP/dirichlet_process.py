import numpy as np
import nltk
from nltk.corpus import brown
import dirichlet as diri
import matplotlib.pyplot as plt

class draw:
    """A draw from a diriclet process """
    cnts = []
    theta = []
    alpha = 0.1

class drawSmall:
    """A draw from a diriclet process only with relatively small number of words"""
    cnts = []
    theta = []

    def CRP(self, data):
        assert(self.alpha>0)
        for i in range(len(data)):
            #print(repr(i)+'\t')
            post = self.posterior(data[i])
            table = np.random.multinomial(1, post)
            if table[-1] == 1:       #a new table is organized and guide the customer
                self.cnts.append(1)
                self.theta.append(self.baseDist.sample())
            else:                    #guide the customer to the prefered table and exit while
                self.cnts[int(table.nonzero()[0])] += 1

    def __init__(self, alpha=0.1, noWords=None, baseDist=None):
        self.alpha = alpha
        if noWords != None:
            self.baseDist = diri.dirichletDist([1]*noWords)
        elif baseDist != None:
            self.baseDist = baseDist
        else:
            assert(False)

    def noWords(self):
        return self.baseDist.dim()

    def noClusters(self):
        assert(len(self.cnts) == len(self.theta))
        return len(self.cnts)

    def posterior(self, sample):
        """sample: index of the observed word"""
        assert(sample in range(self.noWords()))
        posterior=np.zeros(self.noClusters()+1)
        posterior[:-1] = np.array([t[sample] for t in self.theta])* np.array(self.cnts)
        posterior[-1] = self.alpha * self.baseDist.Zpost(sample)
        posterior = posterior/np.sum(posterior)
        assert(np.sum(posterior)-1<1e-10)
        return posterior

    def prior(self):
        cnts = np.array(self.cnts, dtype='float')
        return cnts/np.sum(cnts)

def extractData():
    """
    extract words from brown corpus to compose a vector expression like
    [0,1,3,1,2,0]
    """
    news_text = brown.words(categories="news")
    fdist = nltk.FreqDist([w.lower() for w in news_text])
    words = fdist.samples()[10:6000:40]
    data = [w for w in news_text if w in words]
    return  [words.index(d) for d in data]


def main():
    #diri.npSampleDirichlet(1,(10,20))
    draw = drawSmall(alpha=10, noWords=150)
    data = extractData()
    for i in range(100):
        print('Iteration '+repr(i))
        print('noClusters '+repr(draw.noClusters()))
        print(repr(draw.prior()))
        draw.CRP(data)
    return draw

if __name__=="__main__":
    main()





