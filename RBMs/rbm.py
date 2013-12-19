import __builtin__ as base
from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

from basics import *

import MNIST

class RBM(object):
    def __init__(self, M, N, batchsz=100):
        self.W = 0.01*randn(M,N)
        self.a = 0.01*randn(M)
        self.b = 0.01*randn(N)
        self.vW = zeros(self.W.shape)
        self.va = zeros(self.a.shape)
        self.vb = zeros(self.b.shape)
        self.batchsz = batchsz
        self.algorithm = None
        self.epoch = 0
        self.particles = None
        self.CDN = 1
        self.sparsity = {'strength': 0., 'target': 0.}
        self.shape = (M,N)

    #made it a function because sometimes varing leraning rate is better
    def lrate(self):
        return 0.03
    def drate(self):
        return 1.
    def mom(self):
        if self.epoch<=5:
            return 0
        else:
            return 0.9

    def lrates(self):
        return (self.lrate(), self.mom(), self.drate())

    def initWithData(self, data):
        pass

    def processDat(self, data):
        datasz = data.shape[0]
        assert(mod(datasz ,self.batchsz)==0)
        noBatches = datasz/self.batchsz
        idcs = permutation(datasz).reshape((noBatches, self.batchsz))
        for bid in xrange(noBatches):
            yield data[idcs[bid]]

    #for PCD/CD selection
    def setAlgorithm(self, name):
        assert(self.algorithm == None)
        assert(name in ['CD', 'PCD'])
        self.algorithm = name
        self.particles = rand(self.batchsz, self.shape[1])>0.8

    #training
    def train(self, data, epochs):
        plt.ion()
        plt.hold(False)
        assert(data.shape[1]==self.shape[1])
        assert(self.algorithm != None)
        for epc in xrange(epochs):
            self.epoch = epc
            print('epoch:' + repr(epc))
            self.sweepAcrossData(self.processDat(data))
            #verify
            img = zeros((280,280))
            for i in xrange(10):
                for j in xrange(10):
                    img[i*28:(i+1)*28, j*28:(j+1)*28] = self.W.reshape((10,10,28,28))[i,j]
            plt.imshow(img, animated=True)
            plt.draw()
        return self

    def sweepAcrossData(self,data):
        strength, target = self.sparsity.values()
        batchsz = float(self.batchsz)
        lrate, mom, drate = self.lrates()
        particles = self.particles
        #main part of the training
        for item in data: ##sampled data vectors
            eh = self.expectH(item)          #expected H vectors
            dW = dot(eh.T, item)/batchsz
            da = mean(eh,axis=0)
            db = mean(item,axis=0)
            #sparsity
            da += strength*(target-da) 
            if self.algorithm == 'CD':
                particles = item
            for cdCount in xrange(self.CDN):
                hv, eh = self.sampleH(particles)
                particles, ev = self.sampleV(hv)
            eh = self.expectH(particles)
            dW -= dot(eh.T, particles)/batchsz
            da -= mean(eh,axis=0)
            db -= mean(ev,axis=0)
            self.vW = mom*self.vW + lrate*dW
            self.va = mom*self.va + lrate*da
            self.vb = mom*self.vb + lrate*db
            self.W = drate*self.W + self.vW
            self.a = drate*self.a + self.va
            self.b = drate*self.b + self.vb
        self.particles = particles
        print(sqrt(sum(self.W*self.W)))
    

class BRBM(RBM):
    """Bernoulli RBMs"""
    def expectH(self, V):
        assert(V.shape[1]==self.shape[1])
        return sigmoid(dot(V,self.W.T)+ self.a)
    def sampleH(self, V):
        eh = self.expectH(V)
        return (eh > rand(*eh.shape), eh)

    def expectV(self, H):
        assert(H.shape[1]==self.shape[0])
        return sigmoid(dot(H,self.W)+ self.b)
    def sampleV(self, H):
        ev = self.expectV(H)
        return (ev > rand(*ev.shape), ev)

class GRBM(RBM):
    """Gaussian RBMs"""
    def __init__(self, M, N, batchsz=100):
        super(GRBM, self).__init__(M=M,N=N,batchsz=batchsz)
        self.sigma = ones(self.shape[1])
    def initWithData(self, data):
        super(GRBM, self).initWithData(data)
        self.sigma = 0.9*std(data, axis=0)
    def expectH(self, V):
        assert(V.shape[1]==self.shape[1])
        return sigmoid(dot(V/self.sigma,self.W.T)+ self.a)
    def sampleH(self, V):
        eh = self.expectH(V)
        return (eh > rand(*eh.shape), eh)

    def expectV(self, H):
        assert(H.shape[1]==self.shape[0])
        return self.sigma*dot(H,self.W)+ self.b
    def sampleV(self, H):
        ev = self.expectV(H)
        return (randn(*ev.shape)*self.sigma+ev, ev)


def testBRBM():
    data = MNIST.data()
    rbm = BRBM(M=100, N=784)
    rbm.CDN = 1
    rbm.setAlgorithm('CD')
    rbm.sparsity['strength'] = 1.0
    rbm.sparsity['target'] = 0.05
    print(rbm.train(data['training']['data'], 10))

def testGRBM():
    data = MNIST.data()
    rbm = GRBM(M=100, N=784)
    #rbm.initWithData(data['training']['data'])
    rbm.sigma = 0.5*rbm.sigma
    rbm.CDN = 1
    rbm.setAlgorithm('CD')
    print(rbm.train(data['training']['data'], 50))

if __name__ == "__main__":
    testGRBM()

