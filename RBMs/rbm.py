import __builtin__ as base
from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

import MNIST

def sigmoid(x):
    return 1/(1+exp(-x))

class RBM(object):
    def __init__(self, M, N, batchsz=100, mom=0.9):
        self.W = 0.01*randn(M,N)
        self.a = 0.01*randn(M)
        self.b = 0.01*randn(N)
        self.vW = zeros(self.W.shape)
        self.va = zeros(self.a.shape)
        self.vb = zeros(self.b.shape)
        self.batchsz = batchsz
        self.algorithm = None
        self.mom = mom
        self.drate = 1.
        self.epoch = 0
        self.__sparseParams = None
        self.particles = None
        self.CDN = 1

    #made it a function because sometimes varing leraning rate is better
    def lrate(self):
        return 0.01

    def M(self):
        return self.W.shape[0]

    def N(self):
        return self.W.shape[1]

    def vParams(self):
        return (self.vW, self.va, self.vb)
    def params(self):
        return (self.W, self.a, self.b)
    def lrates(self):
        return (self.lrate(), self.mom, self.drate)

    def initByData(self, data):
        pass

    def processDat(self, data):
        datasz = data.shape[0]
        assert(mod(datasz ,self.batchsz)==0)
        noBatches = datasz/self.batchsz
        return data[ix_(permutation(datasz))].reshape((noBatches, self.batchsz, self.N()))

    #for PCD/CD selection
    def setAlgorithm(self, name):
        assert(self.algorithm == None)
        assert(name in ['CD', 'PCD'])
        self.algorithm = name
        self.particles = rand(self.batchsz, self.N())>0.8

       

class bRBM(RBM):
    """Bernoulli RBMs"""
    def expectH(self, V):
        assert(V.shape[1]==self.N())
        return sigmoid(dot(V,self.W.T)+ self.a)
    def sampleH(self, V):
        eh = self.expectH(V)
        return (eh > rand(*eh.shape), eh)

    def expectV(self, H):
        assert(H.shape[1]==self.M())
        return sigmoid(dot(H,self.W)+ self.b)
    def sampleV(self, H):
        ev = self.expectV(H)
        return (ev > rand(*ev.shape), ev)
    #
    def setSparseParams(self, strength, target):
        self.__sparseParams = (strength, target)

    def sparseRegularizer(self):
        if self.__sparseParams != None:
            sStr, sTgt = self.__sparseParams            
            return lambda da: sStr*(sTgt-da) 
        else:
            return lambda da: 0
            

    def sweepAcrossData(self,data):
        sreg = self.sparseRegularizer()
        batchsz = float(self.batchsz)
        vW,va,vb = self.vParams()
        W, a, b  = self.params()
        lrate, mom, drate = self.lrates()
        particles = self.particles
        #main part of the training
        for item in data: ##sampled data vectors
            #positive phase
            eh = self.expectH(item)          #expected H vectors
            dW = dot(eh.T, item)/batchsz
            da = mean(eh,axis=0)
            db = mean(item,axis=0)
            #sparsity
            da += sreg(da)
            #Switching PCD/CD
            if self.algorithm == 'CD':
                particles = item
            #negative phase
            for cdCount in xrange(self.CDN):
                particles, ev = self.sampleV(self.sampleH(particles)[0])
            #Computing gradient
            eh = self.expectH(particles)
            dW -= dot(eh.T, particles)/batchsz
            da -= mean(eh,axis=0)
            db -= mean(ev,axis=0)
            # update
            vW[:] = mom*vW + lrate*dW
            va[:] = mom*va + lrate*da
            vb[:] = mom*vb + lrate*db
            W[:] = drate*W + vW
            a[:] = drate*a + va
            b[:] = drate*b + vb
        self.particles = particles

    def train(self, data, epochs):
        plt.ion()
        plt.hold(False)
        assert(data.shape[1]==self.N())
        assert(self.algorithm != None)
        self.initByData(data)
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
            #print self.particles
        return self



def main():
    data = MNIST.data()
    rbm = bRBM(M=100, N=784)
    rbm.CDN = 10
    rbm.setAlgorithm('CD')
    rbm.setSparseParams(1.5,0.01)
    print(rbm.train(data[0][0], 50))

if __name__ == "__main__":
    main()

