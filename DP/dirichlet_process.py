import numpy as np
import nltk
import dirichlet as diri
import matplotlib.pyplot as plt
from nltk.corpus import brown

class draw:
    """A draw from a diriclet process """
    cnts = []
    theta = []
    alpha = 0.01

class draw10w:
    """A draw from a diriclet process only with 10 words"""
    cnts = []
    theta = []
    alpha = 0.01
    noWords = 10
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

    def processData(self, data):
        assert(len(data.shape) == 1)
        assert(max(data)<self.noWords)
        assert(min(data)>=0)
        matrix = np.arange(self.noWords)
        return [element == matrix for element in data] #convert data to 1-of-10 expression

    def posterior(self, samples):
        """samples are in 1-of-10 expression"""
        samples = np.array(samples)
        assert(len(samples.shape) == 1)
        assert(np.max(samples)==1)
        assert(np.min(samples)==0)
        assert(np.sum(samples)==1)
        posterior = [cnt*l for l in [diri.npPDFDirichlet(t, samples+1) for t in self.theta]]
        posterior = [pos/(sum(posterior)+self.alpha) for pos in posterior]
        posterior.append(self.alpha/(sum(posterior)+self.alpha))
        assert(sum(posterior)==1)
        return np.array(posterior)


def main():
    diri.npSampleDirichlet(1,(10,20))

if __name__=="__main__":
    main()





