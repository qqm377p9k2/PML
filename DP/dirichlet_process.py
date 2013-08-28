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
    noClusters = 0;
    cnts = []
    theta = []
    noWords = 150

    def CRP(self, data):
        assert(self.alpha>0)
        data = self.processData(data)
        for i in range(len(data)):
            #print(repr(i)+'\t')
            post = self.posterior(data[i])
            table = np.random.multinomial(1, post)
            if table[-1] == 1:       #a new table is organized and guide the customer
                self.noClusters += 1
                self.cnts.append(1)
                self.theta.append(diri.npSampleDirichlet(1,self.noWords))
            else:                    #guide the customer to the prefered table and exit while
                self.cnts[int(table.nonzero()[0])] += 1

    def __init__(self, alpha=0.1, baseDist):
        self.alpha = alpha
        self.baseDist = baseDist

    def processData(self, data):
        assert(max(data)<self.noWords)
        assert(min(data)>=0)
        matrix = np.arange(self.noWords)
        return [element == matrix for element in data] #convert data to 1-of-K expression

    def posterior(self, samples):
        """samples are in 1-of-K expression"""
        samples = np.array(samples)
        assert(len(samples.shape) == 1)
        assert(np.max(samples)==1)
        assert(np.min(samples)==0)
        assert(np.sum(samples)==1)
        likelihood = [diri.npPDFDirichlet(t, samples+1) for t in self.theta]
        posterior=np.zeros(self.noClusters+1)
        posterior[:-1] = np.array(likelihood)* np.array(self.cnts)
        posterior[-1] = self.alpha
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
    draw = drawSmall(alpha=1e262)
    data = extractData()
    for i in range(10):
        print('Iteration '+repr(i))
        print('noClusters '+repr(draw.noClusters))
        print(repr(draw.prior()))
        draw.CRP(data)
    return draw

if __name__=="__main__":
    main()





