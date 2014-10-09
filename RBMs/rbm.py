import __builtin__ as base
from numpy import *
from Basics.numpy_ import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

from Basics.utils import measuring_speed

import cPickle, gzip

#from basics import *
from basics import binary_expression, logSumExp, logDiffExp, sigmoid

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

    def sampleN(self):
        samples = rand(1000, self.shape[1])>0.8
        for count in xrange(1000):
            samples = self.sampleV(self.sampleH(samples)[0])[0]
        return samples

    def sample(self, N=100):
        samples = zeros((N, self.shape[1]))
        samples[0] = rand(1, self.shape[1])>0.5
        hsamples=zeros((N, self.shape[0]))
        for i in xrange(1,N):
            hsamples[i] = self.sampleH(samples[i-1][newaxis, :])[0]
            samples[i] = self.sampleV(hsamples[i][newaxis, :])[0]
        return samples, hsamples

    def set_n(self, n):
        u,s,v = LA.svd(asarray(self.W, dtype='>d'))
        self.W =  (u*s)[:,:n].dot(v[:n,:])

    def set_temp(self, beta=1.):
        self.W = beta*self.W
        self.a = beta*self.a
        self.b = beta*self.b

    @staticmethod
    def all_configs(N, LBS=15):
        if N > LBS:
            bin_block = binary_expression(xrange(2**LBS), LBS)
            for i in xrange(2**(N-LBS)):
                yield concatenate((binary_expression(i*ones(2**LBS), N-LBS), bin_block), 1)
        else:
            yield binary_expression(xrange(2**N), N)

def show_samples(rbm, N=100, verbose=False, hidden=False):
    import DBNs.img_utils
    v,h = rbm.sample(N)
    image = DBNs.img_utils.tile_raster_images(v, (28,28), (int(ceil(float(N)/50)),50))
    if hidden:
        plt.imshow(h, cmap=plt.cm.binary)
        plt.show()
    if verbose:
        next = zeros(v.shape)
        next[1:] = v[:-1]
        print (v!=next).sum(axis=1)
    plt.imshow(255-image, cmap=plt.cm.binary)
    plt.show()
    return v,h


def test_rbmsMFT(w=1.5, T=1000):
    rbm = BRBM(M=2, N=128)
    rbm.W = w*ones(rbm.W.shape)
    rbm.b[:] = 0
    rbm.a[:] = 0
    v,h = rbm.sample(T)
    plt.imshow(v, cmap=plt.cm.binary)
    plt.show()
    

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

    def fe(self, v):
        W, a, b = self.W, self.a, self.b
        axis = len(v.shape)-1
        return -(v*b).sum(axis=axis) - log(1+exp(dot(v,W.T)+ a)).sum(axis=axis) 

    def exact_grad(self, data):
        assert(self.N < 20)
        LBS = 12
        W, a, b = self.W, self.a, self.b
        #positive grad
        batchsz = float(self.batchsz)
        dWpos, dapos, dbpos = zeros(W.shape), zeros(a.shape), zeros(b.shape)
        for item in data: ##sampled data vectors
            eh = self.expectH(item)          #expected H vectors
            dWpos += dot(eh.T, item)/batchsz
            dapos += mean(eh,axis=0)
            dbpos += mean(item,axis=0)
        dWpos /= data.shape[0]
        dapos /= data.shape[0]
        dbpos /= data.shape[0]
        #negative grad
        dWneg, daneg, dbneg = zeros(W.shape), zeros(a.shape), zeros(b.shape)
        logZ = zeros(max(2**(self.N-LBS), 1))
        for i,c in enumerate(RBM.all_configs(self.N, LBS=12)):
            eh = self.expectH(c)
            unpv = exp(-self.fe(c)) #unnormalized p(v)
            dWneg += dot(eh.T*unpv, c)
            daneg += (eh*unph[:, newaxis]).sum(axis=0)
            dbneg += (c*unph[:, newaxis]).sum(axis=0)
            logZ[i] = logSumExp(-self.fe(c))
        logZ = logSumExp(logZ)
        dWneg /= exp(logZ)
        daneg /= exp(logZ)
        dbneg /= exp(logZ)
        return (dWpos, dapos, dbpos), (dWneg, daneg, dbneg), logZ

    def AIS(self, betaseq, N=100, base_rbm=None, mode='logZ'):
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
        logZB_est_bounds = (logDiffExp(asarray([log(3)+logstd_AIS, r_AIS]))[0]+logZA,
                            logSumExp( asarray([log(3)+logstd_AIS, r_AIS]))   +logZA)
        if mode == 'logZ':
            return logZB, logZB_est_bounds
        else:
            return (hid, vis), logZB, logZB_est_bounds

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
        

def show_samples_GRBM2d(N=100, verbose=False, hidden=False):
    rbm = GRBM(2,2)
    theta = np.pi*0.45
    rbm.W = 5*asarray([[1,0],[np.cos(theta), np.sin(theta)]])
    rbm.b = -rbm.W.sum(axis=0)/2.

    v,h = rbm.sample(N)
    v = v.reshape(v.shape[0]*v.shape[1], v.shape[2])
    plt.scatter(v[:,0], v[:,1])

    ext = ranges(v.T)
    print ext.reshape(4)
    X, Y = np.meshgrid(*mesh(ext))
    XY = npcat((X[:, :, newaxis], Y[:, :, newaxis]), axis=2)
    plt.imshow(np.exp(-rbm.fe(XY)), interpolation='bilinear', origin='lower', cmap=cm.gray, extent=ext.reshape(4))
    plt.contour(X, Y, np.exp(-rbm.fe(XY)))
    plt.show()
    return v,h

def show_fe(rbm, ext=[[-10, 10], [-10,10]]):
    ext = asarray(ext)
    X, Y = np.meshgrid(*mesh(ext))
    XY = npcat((X[:, :, newaxis], Y[:, :, newaxis]), axis=2)
    plt.imshow(np.exp(-rbm.fe(XY)), interpolation='bilinear', origin='lower', cmap=cm.gray, extent=ext.reshape(4))
    plt.contour(X, Y, np.exp(-rbm.fe(XY)))
    plt.show()


def make_test_GRBM(dist=5):
    rbm = GRBM(2,2)
    theta = np.pi*0.45
    rbm.W = dist*asarray([[1,0],[np.cos(theta), np.sin(theta)]])
    rbm.b = -rbm.W.sum(axis=0)/2.
    return rbm

def make_test_GRBM4(dist=3):
    rbm = GRBM(2,2)
    theta = np.pi*0.4
    rbm.W = dist*asarray([[1,0],[np.cos(theta), np.sin(theta)]])
    rbm.b = -rbm.W.sum(axis=0)/2.
    return rbm

def make_test_GRBM3():
    rbm = GRBM(20,2)
    rbm.W *= 50
    rbm.b = -rbm.W.sum(axis=0)/2.
    return rbm

def make_test_GRBM2(dist=4):
    rbm = GRBM(2,2)
    theta = np.pi*0.4
    rbm.W = dist*asarray([[1,0],[np.cos(theta), np.sin(theta)]])
    rbm.b = -rbm.W.sum(axis=0)/2.
    return rbm


def show_GRBM_fe(beta=1., N=100):
    rbm = GRBM(2,2)
    theta = np.pi*0.45
    rbm.W = 5*asarray([[1,0],[np.cos(theta), np.sin(theta)]])
    rbm.b = -rbm.W.sum(axis=0)/2.

    v,h = rbm.sample(N)
    print v.shape
    v = v.reshape(v.shape[0]*v.shape[1], v.shape[2])

    ext = ranges(v.T)
    print ext.reshape(4)
    X, Y = np.meshgrid(*mesh(ext))
    XY = npcat((X[:, :, newaxis], Y[:, :, newaxis]), axis=2)
    fmap = np.exp(-(1-beta)*(XY**2).sum(axis=2) -beta*rbm.fe(XY))
    plt.imshow(fmap, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=ext.reshape(4))
    plt.contour(X, Y, fmap)
    plt.show()
    return v,h, rbm
    
def plot_GM_pdf(beta=[1.]):
    x = arange(-10,10, 0.1)
    ratio = [0.7, 0.1, 0.2]
    loc = [-7, 7, 3]
    m = (asarray(ratio)*asarray(loc)).mean()
    print m
    #pdf = GM(x, ratio, loc)
    for b in beta:
        #plt.plot(x, exp(-(1-b)*0.5*(x-m)**2)*(pdf**b))
        plt.plot(x, annealedGM(x, ratio, loc, (m, 1.), b))
    plt.show()


def plot_GM_pdf2(beta=[1.]):
    x = arange(-10,10, 0.1)
    ratio = [0.7, 0.1, 0.2]
    loc = [-7, 7, 3]
    m = (asarray(ratio)*asarray(loc)).mean()
    print m
    for b in beta:
        #plt.plot(x, exp(-(1-b)*0.5*((x-m)/5)**2)*(pdf**b))
        plt.plot(x, annealedGM(x, ratio, loc, (m, 5.), b))
    plt.show()

def GM(x, ratio, mu):
    return asarray([r*np.exp(-0.5*(x-m)**2)/np.sqrt(2*np.pi) for r, m in zip(ratio, mu)]).sum(axis=0)

def annealedGM(x, ratio, mu, base_param, beta):
    mb, sb = base_param
    Z = 0
    for r, mm in zip(ratio, mu):
        m = ((1-beta)*mb + beta*mm*sb**2)/((1-beta) + beta*sb**2)
        s = sb/np.sqrt((1-beta) + beta*sb**2)
        Z += r*np.sqrt(2*np.pi)*s*exp(-0.5*((1-beta)*(mb/sb)**2 + beta*mm**2) + 0.5*(m/s)**2)
    return asarray([r*np.exp(-(beta)*0.5*(x-m)**2)*np.exp(-(1-beta)*0.5*((x-mb)/sb)**2) for r, m in zip(ratio, mu)]).sum(axis=0)/Z

def ranges(vecs):
    return asarray((vecs.min(axis=1), vecs.max(axis=1))).T
def mesh(ranges, N=100.):
    return asarray([arange(min, max, (max-min)/N) for min, max in ranges])

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
    def fe(self, v):
        W, a, b, s = self.W, self.a, self.b, self.sigma
        axis = len(v.shape)-1
        return 0.5*(((v-b)/s)**2).sum(axis=axis) - log(1+exp(dot(v/s,W.T)+ a)).sum(axis=axis) 
    def feh(self, h):
        W, a, b, s = self.W, self.a, self.b, self.sigma
        N = self.N
        return 0.5*((b/s)**2).sum() -0.5*self.N*log(2*np.pi) -log(s).sum() -h.dot(a) - 0.5*(((b+s*h.dot(W))/s)**2).sum(axis=1)
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

    def exact_grad(self, data):
        assert(self.M < 20)
        LBS = 12
        W, a, b = self.W, self.a, self.b
        #positive grad
        batchsz = float(self.batchsz)
        dWpos, dapos, dbpos = zeros(W.shape), zeros(a.shape), zeros(b.shape)
        for item in data: ##sampled data vectors
            eh = self.expectH(item)          #expected H vectors
            dWpos += dot(eh.T, item/self.sigma)/batchsz
            dapos += mean(eh,axis=0)
            dbpos += mean((item-self.b)/(self.sigma**2),axis=0)
        dWpos /= data.shape[0]
        dapos /= data.shape[0]
        dbpos /= data.shape[0]
        #negative grad
        dWneg, daneg, dbneg = zeros(W.shape), zeros(a.shape), zeros(b.shape)
        logZ = zeros(max(2**(self.M-LBS), 1))
        for i,c in enumerate(RBM.all_configs(self.M, LBS=12)):
            ev = self.expectV(c)
            unph = exp(-self.feh(c)) #unnormalized p(h)
            dWneg += dot(c.T*unph, ev/self.sigma)
            daneg += (c*unph[:, newaxis]).sum(axis=0)
            dbneg += (((ev-self.b)/(self.sigma**2))*unph[:, newaxis]).sum(axis=0)
            logZ[i] = logSumExp(-self.feh(c))
        #logZ = logSumExp(asarray([logSumExp(-self.feh(c)) for c in RBM.all_configs(self.M, LBS=LBS)]))
        logZ = logSumExp(logZ)
        dWneg /= exp(logZ)
        daneg /= exp(logZ)
        dbneg /= exp(logZ)
        return (dWpos, dapos, dbpos), (dWneg, daneg, dbneg), logZ

    def exact_natgrad(self, data):
        assert(self.M < 20)
        LBS = 12
        W, a, b = self.W, self.a, self.b
        #positive grad
        batchsz = float(self.batchsz)
        dWpos, dapos, dbpos = zeros(W.shape), zeros(a.shape), zeros(b.shape)
        for item in data: ##sampled data vectors
            eh = self.expectH(item)          #expected H vectors
            dWpos += dot(eh.T, item/self.sigma)/batchsz
            dapos += mean(eh,axis=0)
            dbpos += mean((item-self.b)/(self.sigma**2),axis=0)
        dWpos /= data.shape[0]
        dapos /= data.shape[0]
        dbpos /= data.shape[0]
        #negative grad
        dWneg, daneg, dbneg = zeros(W.shape), zeros(a.shape), zeros(b.shape)
        nparams = (self.N+1)*(self.M+1)-1
        dfcov, dfmean = zeros((nparams, nparams)), zeros(nparams)
        logZ = zeros(max(2**(self.M-LBS), 1))
        for i,c in enumerate(RBM.all_configs(self.M, LBS=12)):
            ev = self.expectV(c)
            unph = exp(-self.feh(c)) #unnormalized p(h)
            dWneg += dot(c.T*unph, ev/self.sigma)
            daneg += (c*unph[:, newaxis]).sum(axis=0)
            dbneg += (((ev-self.b)/(self.sigma**2))*unph[:, newaxis]).sum(axis=0)
            #dfcov += 
            #dfmean+= 
            logZ[i] = logSumExp(-self.feh(c))
        #logZ = logSumExp(asarray([logSumExp(-self.feh(c)) for c in RBM.all_configs(self.M, LBS=LBS)]))
        logZ = logSumExp(logZ)
        dWneg /= exp(logZ)
        daneg /= exp(logZ)
        dbneg /= exp(logZ)
        return (dWpos, dapos, dbpos), (dWneg, daneg, dbneg), logZ

    def AIS(self, betaseq, N=100, base_params=None, mode='logZ'):
        if base_params is None:
            base_params = self.cov(1.0)
        M = self.M
        bA, sA = base_params[0], np.sqrt(diag(base_params[1]))
        WB, aB, bB, sB = self.W, self.a, self.b, self.sigma
        logZA =M*log(2) + 0.5*self.N*log(2*np.pi) + log(sA).sum()

        prob = tile(bA, [N, 1])
        vis = randn(*prob.shape)*sA + prob

        logw =  0.5*(((vis-bA)/sA)**2).sum(axis=1) - M*log(2)

        for beta in betaseq[1:-1]:
            logw += -0.5*((1-beta)*((vis-bA)/sA)**2 + beta*((vis-bB)/sB)**2).sum(axis=1) + ReL(beta*((vis/sB).dot(WB.T)+aB)).sum(axis=1)
            prob = sigmoid(beta*((vis/sB).dot(WB.T)+aB))
            hid  = prob > rand(*prob.shape)
            prob = (beta*(hid.dot(WB/sB)+bB/(sB**2)) + (1-beta)*bA/(sA**2))/(beta/(sB**2) + (1-beta)/(sA**2))
            vis = randn(*prob.shape)/np.sqrt(beta/(sB**2) + (1-beta)/(sA**2)) + prob
            logw -= -0.5*((1-beta)*((vis-bA)/sA)**2 + beta*((vis-bB)/sB)**2).sum(axis=1) + ReL(beta*((vis/sB).dot(WB.T)+aB)).sum(axis=1)
        logw += -0.5*(((vis-bB)/sB)**2).sum(axis=1) + ReL((vis/sB).dot(WB.T)+aB).sum(axis=1)

        r_AIS = logSumExp(logw) - log(N)
        logZB = r_AIS + logZA 

        ESS = exp(2*logSumExp(logw)-logSumExp(2*logw))

        meanlogw = logw.mean()
        logstd_AIS = log(std(exp(logw-meanlogw))) + meanlogw -log(N)/2
        logZB_est_bounds = (logDiffExp(asarray([log(3)+logstd_AIS, r_AIS]))[0]+logZA,
                            logSumExp( asarray([log(3)+logstd_AIS, r_AIS]))   +logZA)
        if mode == 'logZ':
            return logZB, logZB_est_bounds, ESS
        else:
            return (hid, vis), logZB, logZB_est_bounds

    def AIS_mean(self, betaseq, N=100, base_params=None, mode='logZ'):
        if base_params is None:
            base_params = self.cov(1.0)
        base_params = base_params[0], diag(self.sigma)**2
        return self.AIS(betaseq, N, base_params, mode)


    def AIS_debug(self, betaseq, N=100, base_params=None, mode='logZ', pause_points=None, ext=([-10,10],[-10,10]), save=None):
        if base_params is None:
            base_params = self.cov(1.0)
        if pause_points==None:
            pause_points = (len(betaseq)-3)*ones(12)
            pause_points[:-1] = arange(0, len(betaseq), int(len(betaseq)/10.))
        pause_points = asarray(pause_points, dtype=int)
        print zip(betaseq[pause_points], pause_points)
        M = self.M
        bA, sA = base_params[0], np.sqrt(diag(base_params[1]))
        WB, aB, bB, sB = self.W, self.a, self.b, self.sigma
        logZA =M*log(2) + 0.5*self.N*log(2*np.pi) + log(sA).sum()
        print 'logZA:', logZA

        prob = tile(bA, [N, 1])
        vis = randn(*prob.shape)*sA + prob

        logw =  0.5*(((vis-bA)/sA)**2).sum(axis=1) - M*log(2)


        assert(self.N==2)
        ext = asarray(ext)
        X, Y = np.meshgrid(*mesh(ext))
        XY = npcat((X[:, :, newaxis], Y[:, :, newaxis]), axis=2)

        for i, beta in enumerate(betaseq[1:-1]):
            if i in pause_points:
                print beta
                fmap = np.exp(-0.5*((1-beta)*((XY-bA)/sA)**2 + beta*((XY-bB)/sB)**2).sum(axis=2) + ReL(beta*(np.tensordot((XY/sB), WB.T, axes=[2,0])+aB)).sum(axis=2))
                fig = plt.figure()
                ax = fig.add_subplot(111, aspect='equal')
                plt.imshow(fmap, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=ext.reshape(4))
                plt.scatter(vis[:,0], vis[:,1], c='cyan', s=30)
                ax.set_xlim(*ext[0])
                ax.set_ylim(*ext[1])
                ax.set_axis_off()
                if save:
                    plt.savefig(save+'%g.png'%beta, bbox_inches='tight')
                else:
                    plt.show()
            logw += -0.5*((1-beta)*((vis-bA)/sA)**2 + beta*((vis-bB)/sB)**2).sum(axis=1) + ReL(beta*((vis/sB).dot(WB.T)+aB)).sum(axis=1)
            prob = sigmoid(beta*((vis/sB).dot(WB.T)+aB))
            hid  = prob > rand(*prob.shape)
            prob = (beta*(hid.dot(WB/sB)+bB/(sB**2)) + (1-beta)*bA/(sA**2))/(beta/(sB**2) + (1-beta)/(sA**2))
            vis = randn(*prob.shape)/np.sqrt(beta/(sB**2) + (1-beta)/(sA**2)) + prob
            logw -= -0.5*((1-beta)*((vis-bA)/sA)**2 + beta*((vis-bB)/sB)**2).sum(axis=1) + ReL(beta*((vis/sB).dot(WB.T)+aB)).sum(axis=1)
        logw += -0.5*(((vis-bB)/sB)**2).sum(axis=1) + ReL((vis/sB).dot(WB.T)+aB).sum(axis=1)

        r_AIS = logSumExp(logw) - log(N)
        logZB = r_AIS + logZA 

        ESS = exp(2*logSumExp(logw)-logSumExp(2*logw))

        meanlogw = logw.mean()
        logstd_AIS = log(std(exp(logw-meanlogw))) + meanlogw -log(N)/2
        logZB_est_bounds = (logDiffExp(asarray([log(3)+logstd_AIS, r_AIS]))[0]+logZA,
                            logSumExp( asarray([log(3)+logstd_AIS, r_AIS]))   +logZA)
        if mode == 'logZ':
            return logZB, logZB_est_bounds, ESS, r_AIS
        else:
            return (hid, vis), logZB, logZB_est_bounds


    def AIS_cov(self, betaseq, base_params=None, N=100, mode='logZ'):
        if base_params is None:
            base_params = self.cov(1.1)
        M = self.M
        mu, cov = base_params
        covinv = linalg.inv(cov)
        #coveig = linalg.eigh(cov)
        WB, aB, bB, sB = self.W, self.a, self.b, self.sigma
        LA = linalg.inv(cov-np.diag(sB**2))
        LAeig = linalg.eigh(LA)[0]
        print LAeig
        assert((LAeig>0).all())

        covxv = linalg.inv(diag(1/sB**2)+LA)

        logZA =M*log(2) + 0.5*self.N*log(2*np.pi) + 0.5*log(linalg.det(cov))

        vis = random.multivariate_normal(mu, cov, size=N)
        logw =  0.5*(((vis-mu).dot(covinv))*(vis-mu)).sum(axis=1) - M*log(2)

        for beta in betaseq[1:-1]:
            logw += -0.5*((1-beta)*((vis-mu).dot(covinv))*(vis-mu)).sum(axis=1) -0.5*(beta*((vis-bB)/sB)**2).sum(axis=1) + ReL(beta*((vis/sB).dot(WB.T)+aB)).sum(axis=1)

            prob = sigmoid(beta*((vis/sB).dot(WB.T)+aB))
            hid  = prob > rand(*prob.shape)
            prob = (vis/sB**2+mu.dot(LA)).dot(covxv)
            x = random.multivariate_normal(zeros(self.N), covxv/(1-beta), N) + prob

            prob = (beta*(hid.dot(sB*WB)+bB) + (1-beta)*x)
            vis = randn(*prob.shape)*sB + prob
            logw -= -0.5*((1-beta)*((vis-mu).dot(covinv))*(vis-mu)).sum(axis=1) -0.5*(beta*((vis-bB)/sB)**2).sum(axis=1) + ReL(beta*((vis/sB).dot(WB.T)+aB)).sum(axis=1)
        logw += -0.5*(((vis-bB)/sB)**2).sum(axis=1) + ReL((vis/sB).dot(WB.T)+aB).sum(axis=1)

        r_AIS = logSumExp(logw) - log(N)
        logZB = r_AIS + logZA 

        ESS = exp(2*logSumExp(logw)-logSumExp(2*logw))

        meanlogw = logw.mean()
        logstd_AIS = log(std(exp(logw-meanlogw))) + meanlogw -log(N)/2
        logZB_est_bounds = (logDiffExp(asarray([log(3)+logstd_AIS, r_AIS]))[0]+logZA,
                            logSumExp( asarray([log(3)+logstd_AIS, r_AIS]))   +logZA)
        if mode == 'logZ':
            return logZB, logZB_est_bounds, ESS
        else:
            return (hid, vis), logZB, logZB_est_bounds


    def AIS_cov_debug(self, betaseq, base_params=None, N=100, mode='logZ', pause_points=None, ext=([-10,10],[-10,10]), save=None):
        if base_params is None:
            base_params = self.cov(1.1)
        if pause_points==None:
            pause_points = (len(betaseq)-3)*ones(12)
            pause_points[:-1] = arange(0, len(betaseq), int(len(betaseq)/10.))
        M = self.M
        mu, cov = base_params
        covinv = linalg.inv(cov)
        #coveig = linalg.eigh(cov)
        WB, aB, bB, sB = self.W, self.a, self.b, self.sigma
        LA = linalg.inv(cov-np.diag(sB**2))
        LAeig = linalg.eigh(LA)[0]
        print LAeig
        assert((LAeig>0).all())

        covxv = linalg.inv(diag(1/sB**2)+LA)

        logZA =M*log(2) + 0.5*self.N*log(2*np.pi) + 0.5*log(linalg.det(cov))
        print 'logZA:', logZA

        vis = random.multivariate_normal(mu, cov, size=N)
        logw =  0.5*(((vis-mu).dot(covinv))*(vis-mu)).sum(axis=1) - M*log(2)

        assert(self.N==2)
        ext = asarray(ext)
        X, Y = np.meshgrid(*mesh(ext))
        XY = npcat((X[:, :, newaxis], Y[:, :, newaxis]), axis=2)

        for i, beta in enumerate(betaseq[1:-1]):
            if i in pause_points:
                print beta
                #fmap = np.exp(-0.5*((1-beta)*((XY-bA)/sA)**2 + beta*((XY-bB)/sB)**2).sum(axis=2) + ReL(beta*(np.tensordot((XY/sB), WB.T, axes=[2,0])+aB)).sum(axis=2))
                fig = plt.figure()
                ax = fig.add_subplot(111, aspect='equal')
                fmap = np.exp(-0.5*((1-beta)*(np.tensordot((XY-mu), covinv, axes=[2,0]))*(XY-mu)).sum(axis=2) -0.5*(beta*((XY-bB)/sB)**2).sum(axis=2) + ReL(beta*(np.tensordot((XY/sB),WB.T, axes=[2,0])+aB)).sum(axis=2))
                plt.imshow(fmap, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=ext.reshape(4))
                plt.scatter(vis[:,0], vis[:,1], c='cyan', s=30)
                ax.set_xlim(*ext[0])
                ax.set_ylim(*ext[1])
                ax.set_axis_off()
                if save:
                    plt.savefig(save+'%g.png'%beta, bbox_inches='tight')
                else:
                    plt.show()
            logw += -0.5*((1-beta)*((vis-mu).dot(covinv))*(vis-mu)).sum(axis=1) -0.5*(beta*((vis-bB)/sB)**2).sum(axis=1) + ReL(beta*((vis/sB).dot(WB.T)+aB)).sum(axis=1)

            prob = sigmoid(beta*((vis/sB).dot(WB.T)+aB))
            hid  = prob > rand(*prob.shape)
            prob = (vis/sB**2+mu.dot(LA)).dot(covxv)
            x = random.multivariate_normal(zeros(self.N), covxv/(1-beta), N) + prob

            prob = (beta*(hid.dot(sB*WB)+bB) + (1-beta)*x)
            vis = randn(*prob.shape)*sB + prob
            logw -= -0.5*((1-beta)*((vis-mu).dot(covinv))*(vis-mu)).sum(axis=1) -0.5*(beta*((vis-bB)/sB)**2).sum(axis=1) + ReL(beta*((vis/sB).dot(WB.T)+aB)).sum(axis=1)
        logw += -0.5*(((vis-bB)/sB)**2).sum(axis=1) + ReL((vis/sB).dot(WB.T)+aB).sum(axis=1)

        r_AIS = logSumExp(logw) - log(N)
        logZB = r_AIS + logZA 

        ESS = exp(2*logSumExp(logw)-logSumExp(2*logw))

        meanlogw = logw.mean()
        logstd_AIS = log(std(exp(logw-meanlogw))) + meanlogw -log(N)/2
        logZB_est_bounds = (logDiffExp(asarray([log(3)+logstd_AIS, r_AIS]))[0]+logZA,
                            logSumExp( asarray([log(3)+logstd_AIS, r_AIS]))   +logZA)
        if mode == 'logZ':
            return logZB, logZB_est_bounds, ESS, r_AIS
        else:
            return (hid, vis), logZB, logZB_est_bounds

    def computeZ(self):
        assert(self.M < 30)
        logZ = logSumExp(asarray([logSumExp(-self.feh(c)) for c in RBM.all_configs(self.M, LBS=12)]))
        return logZ


    def computeMoments(self):
        assert(self.M < 30)
        logZ = logSumExp(asarray([logSumExp(-self.feh(c)) for c in RBM.all_configs(self.M, LBS=12)]))
        return logZ

    def sample(self, N=100, nbatches=100, radius=5., v0=None):
        if v0 is None:
            v0 = radius*randn(nbatches, self.shape[1])
        v = zeros((N, nbatches, self.shape[1]))
        h = zeros((N, nbatches, self.shape[0]))
        v[0] = v0
        for i in xrange(1,N):
            h[i] = self.sampleH(v[i-1])[0]
            v[i] = self.sampleV(h[i])[0]
        return v, h

    def cov(self, scale=1.05, N=100, T=5000, radius=5.):
        v, h = self.sample(N=T, nbatches=N, radius=radius)
        N,nbatches,visnum = v.shape
        v = v[100:].reshape((N-100)*nbatches, visnum)
        return v.mean(axis=0), scale*cov(v.T)

    def cov2(self, scale=1.05, nbatches=100, T=5000, tbatches=100, radius=5.):
        v, h = self.sample(N=100, nbatches=nbatches, radius=radius)
        visnum = self.N
        mvv = zeros((visnum, visnum))
        mv  = zeros(visnum)
        for i in xrange(T/tbatches):
            v, h = self.sample(N=tbatches, nbatches=nbatches, v0=v[-1])
            tmp = v.reshape(tbatches*nbatches, visnum)
            mvv += tmp.T.dot(tmp)
            mv  += tmp.sum(axis=0)
        mv  /= (T*nbatches-1)
        mvv /= (T*nbatches-1)
        return mv, scale*(mvv - mv[:,newaxis].dot(mv[newaxis,:]))


    def sample_geometric_avr_cov(self, beta, base_params=None, nbatches=100, T=1000):
        if base_params is None:
            base_params = self.cov(1.1)
        mu, cov = base_params
        covinv = linalg.inv(cov)
        #coveig = linalg.eigh(cov)
        WB, aB, bB, sB = self.W, self.a, self.b, self.sigma
        LA = linalg.inv(cov-np.diag(sB**2))
        LAeig = linalg.eigh(LA)[0]
        print LAeig
        assert((LAeig>0).all())

        covxv = linalg.inv(diag(1/sB**2)+LA)

        v = zeros((T, nbatches, self.N))
        x = zeros((T, nbatches, self.N))
        h = zeros((T, nbatches, self.M))
        v[0] = random.randn(nbatches, self.N)

        for i in xrange(T-1):
            prob = sigmoid(beta*((v[i]/sB).dot(WB.T)+aB))
            h[i]  = prob > rand(*prob.shape)
            prob = (v[i]/sB**2+mu.dot(LA)).dot(covxv)
            x[i] = random.multivariate_normal(zeros(self.N), covxv/(1-beta), nbatches) + prob

            prob = (beta*(h[i].dot(sB*WB)+bB) + (1-beta)*x[i])
            v[i+1] = randn(*prob.shape)*sB + prob

        return v,x,h


def base_GRBM_for(data, batchsz=100, debug=False):
    N = data.shape[1]
    rbm = BRBM(M=1, N=N)
    rbm.W[:] = 0
    rbm.a[:] = 0
    rbm.b = data.mean(axis=0)
    rbm.sigma = data.std(axis=0)
    return rbm


def load_RBM_mat(dir='M20', data=True):
    import scipy.io as sio
    name = dir+'/rbm.mat'
    params = sio.loadmat(name)
    W = params['W']
    a = (params['a'].T)[0]
    b = (params['b'].T)[0]
    sigma = (params['sigma'].T)[0]
    rbm = GRBM(M=W.shape[0], N=W.shape[1])
    rbm.W = asarray(W, dtype='>d')
    rbm.a = asarray(a, dtype='>d')
    rbm.b = asarray(b, dtype='>d')
    rbm.sigma = asarray(sigma[:-1], dtype='>d')
    data = None
    if data:
        name = dir+'/train.mat'
        data = sio.loadmat(name)
        data = asarray(data['train'], dtype='>d')
        return rbm, data
    else:
        return rbm


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
    pass

