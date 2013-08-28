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
    """A draw from a diriclet process only with relative small number of words"""
    cnts = []
    theta = []
    alpha = 0.1
    noWords = 150
    def CRP(self, data):
        assert(self.alpha>0)
        data = self.processData(data)
        for i in range(len(data)):
            while True:
                table = np.random.multinominal(self.posterior(data[i]))
                if table[-1] == 1:       #a new table is organized
                    self.cnts.append(0)
                    self.theta.append(diri.npSampleDirichlet(1,self.noWords))
                else:                    #guide the customer to the prefered table and exit while
                    self.cnts += table[:-1]
                    break

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def processData(self, data):
        assert(len(data.shape) == 1)
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
        posterior = [cnt*l for l in likelihood]
        posterior = [pos/(sum(posterior)+self.alpha) for pos in posterior]
        posterior.append(self.alpha/(sum(posterior)+self.alpha))
        assert(sum(posterior)==1)
        return np.array(posterior)

def extractData(self):
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
    draw = draw10w()
    data = draw.processData(draw.extractData())
    draw.CRP(data)

if __name__=="__main__":
    main()





