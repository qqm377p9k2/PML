import __builtin__ as base
from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

import cPickle, gzip

from basics import *

from variedParam import *
from Data import *
import MNIST

def emptyMonitor(rbm, data):
    pass

def emptyLogger(mode='log', **kwargs):
    if mode=='init':
        pass
    elif mode=='log':
        pass

def genLogger(output, interval=10):
    def logger(mode='log', **kwargs):
        if mode=='init':
            info = {'nSamples': kwargs['epochs']/interval}
            cPickle.dump(info, output)
            cPickle.dump(kwargs['data'], output)
            cPickle.dump(kwargs['rbm'], output)
        elif mode=='logging':
            if (0==mod(kwargs['rbm'].epoch, interval)):
                cPickle.dump(kwargs['rbm'], output)
    return logger

class RBM(object):
    def __init__(self, M=None, N=None, shape=None, batchsz=100, init_lr=0.005, CDN=1):
        if shape:
            assert((M is None) and (N is None))
            M, N = shape
        self.shape = (M,N)
        self.N = N
        self.M = M

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
        self.CDN = CDN
        self.sparsity = {'strength': 0., 'target': 0.}
        #learning rates
        self.lrate = variedParam(init_lr)
        self.drate = variedParam(1.0)
        self.mom   = variedParam(0.0, [['switchToAValueAt', 5, 0.9]])

    def lrates(self):
        return (self.lrate.value(self.epoch), 
                self.mom.value(self.epoch),
                self.drate.value(self.epoch))

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
        assert(name in ['CD', 'PCD', 'TRUE'])
        self.algorithm = name
        self.particles = rand(self.batchsz, self.shape[1])>0.8

    def activationProb(self, data):
        return mean(self.expectH(data.training.data), axis=0)

    #training
    def train(self, data, epochs,
              monitor=emptyMonitor, logger=emptyLogger):
        assert(data.training.data.shape[1]==self.shape[1])
        assert(self.algorithm != None)
        logger(mode='init', rbm=self, epochs=epochs, data=data)
        for epc in xrange(1,1+epochs):
            self.epoch = epc
            print('epoch:' + repr(epc))
            self.sweepAcrossData(self.processDat(data.training.data))
            monitor(self, data)
            logger(mode='logging', rbm=self)
        return self

    def sample(self):
        samples = rand(1000, self.shape[1])>0.8
        for count in xrange(1000):
            samples = self.sampleV(self.sampleH(samples)[0])[0]
        return samples

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
                particles, ev = self.sampleV(self.sampleH(particles)[0])
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

    def AIS(self, betaseq, N=100, base_rbm=None):
        if not(base_rbm):
            base_rbm = BRBM(M=self.M, N=self.N)
        bA = base_rbm.b
        MA = base_rbm.shape[0]
        WB, aB, bB = self.W, self.a, self.b
        logZA =MA*log(2) + ReL(bA).sum()
        prob = sigmoid(tile(bA, [N, 1]))
        vis = prob > rand(*prob.shape)
        logw = -(vis.dot(bA) + MA*log(2))
        for beta in betaseq[1:-1]:
            logw += (1-beta)*vis.dot(bA) + beta*vis.dot(bB) + ReL(beta*(vis.dot(WB.T)+aB)).sum(axis=1)
            prob = sigmoid(beta*(vis.dot(WB.T)+aB))
            hid  = prob > rand(*prob.shape)
            prob = sigmoid((1-beta)*bA + beta*(hid.dot(WB)+bB))
            vis  = prob > rand(*prob.shape)
            logw -= (1-beta)*vis.dot(bA) + beta*vis.dot(bB) + ReL(beta*(vis.dot(WB.T)+aB)).sum(axis=1)
        logw += vis.dot(bB) + ReL((vis.dot(WB.T)+aB)).sum(axis=1)
        r_AIS = logSumExp(logw) - log(N)
        logZB = r_AIS + logZA 

        meanlogw = logw.mean()
        logstd_AIS = log(std(exp(logw-meanlogw))) + meanlogw -log(N)/2
        logZB_est_bounds = (logSumExp(asarray([log(3)+logstd_AIS, 1./r_AIS]))+logZA,
                            logSumExp(asarray([log(3)+logstd_AIS,    r_AIS]))+logZA)
        return logZB, logZB_est_bounds

def basemodel_for(data, batchsz=100, debug=False):
    #data = MNIST.data()
    N = data.dim
    rbm = BRBM(M=1, N=784)
    rbm.W[:] = 0
    rbm.a[:] = 0
    p = zeros(N)
    nbatches = data.training.size/batchsz
    assert(data.training.size%batchsz == 0)
    batches = data.training.data.reshape((nbatches, batchsz, N))
    batches = batches > random.rand(*batches.shape)
    for batch in batches:
        p += batch.sum(axis=0)
    p /= float(data.training.size)
    if debug:
        plt.plot(p)
        plt.show()
    rbm.b = log(p) - log(1-p)
    rbm.b[isinf(rbm.b)] = -500
    return rbm

def ReL(x):
    ans = log(1+exp(x))
    ans[x>500] = x[x>500]
    return ans

def betaseq(separators, counts):
    assert(separators[0]==0.0 and separators[-1]==1.0)
    assert(all(diff(separators)>0))
    import functools as ft
    import operator as op
    intervals= diff(separators)
    seq = [intv*arange(c)/c for c, intv in zip(counts, intervals)]
    seq = [list(subseq+sep) for subseq, sep in zip(seq, separators[:-1])]
    seq.append([1.])
    seq = ft.reduce(op.add, seq)
    return asarray(seq)
        

class GRBM(RBM):
    """Gaussian RBMs"""
    def __init__(self, M, N, batchsz=100):
        super(GRBM, self).__init__(M=M,N=N,batchsz=batchsz)
        self.sigma = ones(self.shape[1])
    def initWithData(self, data):
        super(GRBM, self).initWithData(data)
        self.sigma = 0.9*std(data.training.data, axis=0)
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
    def sweepAcrossData(self,data):
        strength, target = self.sparsity.values()
        batchsz = float(self.batchsz)
        lrate, mom, drate = self.lrates()
        particles = self.particles
        #main part of the training
        for item in data: ##sampled data vectors
            eh = self.expectH(item)          #expected H vectors
            dW = dot(eh.T, item/self.sigma)/batchsz
            da = mean(eh,axis=0)
            db = mean((item-self.b)/(self.sigma**2),axis=0)
            #sparsity
            da += strength*(target-da) 
            if self.algorithm == 'CD':
                particles = item
            for cdCount in xrange(self.CDN):
                particles, ev = self.sampleV(self.sampleH(particles)[0])
            eh = self.expectH(particles)
            dW -= dot(eh.T, particles/self.sigma)/batchsz
            da -= mean(eh,axis=0)
            db -= mean((ev-self.b)/(self.sigma**2),axis=0)
            self.vW = mom*self.vW + lrate*dW
            self.va = mom*self.va + lrate*da
            self.vb = mom*self.vb + lrate*db
            self.W = drate*self.W + self.vW
            self.a = drate*self.a + self.va
            self.b = drate*self.b + self.vb
        self.particles = particles
        print(sqrt(sum(self.W*self.W)))


def monitorInit():
    plt.ion()
    plt.hold(False)    

def monitor(rbm, data):
    img = zeros((280,280))
    for i in xrange(10):
        for j in xrange(10):
            img[i*28:(i+1)*28, j*28:(j+1)*28] = rbm.W[:100].reshape((10,10,28,28))[i,j]
    plt.subplot(2,1,1)
    plt.imshow(img, animated=True)
    print sort(LA.svd(rbm.W)[1])
    N, M = rbm.shape
    W = concatenate((concatenate((zeros((N,N)), rbm.W), axis=1), 
                     concatenate((rbm.W.T     , zeros((M,M))), axis=1)))
    plt.subplot(2,1,2)
    plt.plot(sort(LA.eigh(W)[0]))
    plt.draw()


def testBRBM():
    monitorInit()
    data = MNIST.data()
    rbm = BRBM(M=500, N=784)
    rbm.CDN = 1
    rbm.setAlgorithm('CD')
    rbm.sparsity['strength'] = 0.0
    rbm.sparsity['target'] = 0.05
    print(rbm.train(data, 10,
                    monitor = monitor))

def testGRBM():
    monitorInit()
    data = MNIST.data()
    rbm = GRBM(M=100, N=784)
    #rbm.initWithData(data['training']['data'])
    rbm.sigma = 0.5*rbm.sigma
    rbm.CDN = 1
    rbm.setAlgorithm('CD')
    print(rbm.train(data, 50,
                    monitor = monitor))

if __name__ == "__main__":
    testBRBM()

