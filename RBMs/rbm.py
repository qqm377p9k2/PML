import __builtin__ as base
from numpy import *
from Basics.numpy_ import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

from Basics.utils import measuring_speed, static_var

from DBNs.img_utils import tile_raster_images
import cPickle, gzip

#from basics import *
from Basics.basics import binary_expression, logSumExp, logDiffExp, sigmoid

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

    def sample(self, v0, T=100, nbatches=100):
        v = zeros((T, nbatches, self.shape[1]))
        h = zeros((T, nbatches, self.shape[0]))
        v[0] = v0
        for i in xrange(1,T):
            h[i] = self.sampleH(v[i-1])[0]
            v[i] = self.sampleV(h[i])[0]
        return v, h

    def sample_old(self, N=100):
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
    def show_filters(self):
        filters = (self.W.reshape(self.M, 3, self.N/3)).transpose(1,0,2)
        image = tile_raster_images((filters[0], filters[1], filters[2], None), (6,6), (int(ceil(float(self.M)/50)),50))
        plt.imshow(image)
        plt.show()


    @staticmethod
    def all_configs(N, LBS=15):
        if N > LBS:
            bin_block = binary_expression(xrange(2**LBS), LBS)
            for i in xrange(2**(N-LBS)):
                yield concatenate((binary_expression(i*ones(2**LBS), N-LBS), bin_block), 1)
        else:
            yield binary_expression(xrange(2**N), N)

def show_samples_BRBM(rbm, N=100, verbose=False, hidden=False):
    import DBNs.img_utils
    v,h = rbm.sample(N, nbatches=1)
    v = v.reshape(v.shape[0], v.shape[2])
    print v.shape
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


def show_samples_GRBM(rbm, N=100, verbose=False, hidden=False):
    import DBNs.img_utils
    v,h = rbm.sample(N, nbatches=1)
    v = v.reshape(v.shape[0], 3, v.shape[2]/3)
    v = v.transpose(1,0,2)
    print v.shape
    image = DBNs.img_utils.tile_raster_images((v[0], v[1], v[2], None), (6,6), (int(ceil(float(N)/50)),50))
    if hidden:
        plt.imshow(h, cmap=plt.cm.binary)
        plt.show()
    if verbose:
        next = zeros(v.shape)
        next[1:] = v[:-1]
        print (v!=next).sum(axis=1)
    plt.imshow(image)
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
        return -(v*b).sum(axis=axis) - ReL(dot(v,W.T)+ a).sum(axis=axis) 

    def feh(self, h):
        W, a, b = self.W, self.a, self.b
        axis = len(h.shape)-1
        return -(h*a).sum(axis=axis) - ReL(dot(h,W)+ b).sum(axis=axis) 

    def computeZ(self):
        assert(self.M < 30)
        logZ = logSumExp(asarray([logSumExp(-self.feh(c)) for c in RBM.all_configs(self.M, LBS=12)]))
        return logZ

    def exact_grad(self, data):
        assert(self.N < 30)
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
            daneg += (eh*unpv[:, newaxis]).sum(axis=0)
            dbneg += (c*unpv[:, newaxis]).sum(axis=0)
            logZ[i] = logSumExp(-self.fe(c))
        logZ = logSumExp(logZ)
        dWneg /= exp(logZ)
        daneg /= exp(logZ)
        dbneg /= exp(logZ)
        return (dWpos, dapos, dbpos), (dWneg, daneg, dbneg), logZ

    def compute_fisher(self):
        ''''
        computes fisher information of an RBM
        '''
        assert(self.N < 20)
        LBS = 12
        batchsz = 2**min(LBS, self.N)
        W, a, b = self.W, self.a, self.b
        #negative grad
        dWneg, daneg, dbneg = zeros(W.shape), zeros(a.shape), zeros(b.shape)
        vhnum,vnum,hnum,paramnum = self.N * self.M, self.N, self.M, (self.N+1)*(self.M+1)-1
        dfecov, dfemean = zeros((paramnum, paramnum)), zeros(paramnum)
        logZ = zeros(max(2**(self.M-LBS), 1))
        dfe_tmp = zeros((batchsz, paramnum))
        def dfe(vis, dfe_tmp):
            dW = -sigmoid(vis.dot(W.T)+a).reshape(batchsz, hnum, 1) * vis.reshape(batchsz, 1, vnum)
            da = -sigmoid(vis.dot(W.T)+a)
            db = -vis
            dfe_tmp[:, :vhnum] = dW.reshape(batchsz, vhnum)
            dfe_tmp[:, vhnum:vhnum+hnum] = da
            dfe_tmp[:, vhnum+hnum:] = db
        for i,c in enumerate(RBM.all_configs(self.N, LBS=LBS)):
            unpv = exp(-self.fe(c)) #unnormalized p(h)
            dfe(c, dfe_tmp)
            dfecov += dot(dfe_tmp.T*unpv, dfe_tmp)
            dfemean+= dot(unpv, dfe_tmp)
            logZ[i] = logSumExp(-self.fe(c))
        #logZ = logSumExp(asarray([logSumExp(-self.feh(c)) for c in RBM.all_configs(self.M, LBS=LBS)]))
        logZ = logSumExp(logZ)
        dfecov /= exp(logZ)
        dfemean/= exp(logZ)
        dfecov -= dfemean[:, newaxis]*dfemean
        return dfecov, logZ

    def exact_natgrad(self, data):
        assert(self.N < 30)
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
        vhnum,vnum,hnum,paramnum = self.N * self.M, self.N, self.M, (self.N+1)*(self.M+1)-1
        dfecov, dfemean = zeros((paramnum, paramnum)), zeros(paramnum)
        logZ = zeros(max(2**(self.M-LBS), 1))
        dfe_tmp = zeros((batchsz, paramnum))
        logZ = zeros(max(2**(self.N-LBS), 1))
        def dfe(vis, dfe_tmp):
            dW = -sigmoid(vis.dot(W.T)+a).reshape(batchsz, hnum, 1) * vis.reshape(batchsz, 1, vnum)
            da = -sigmoid(vis.dot(W.T)+a)
            db = -vis
            dfe_tmp[:, :vhnum] = dW.reshape(batchsz, vhnum)
            dfe_tmp[:, vhnum:vhnum+hnum] = da
            dfe_tmp[:, vhnum+hnum:] = db
        for i,c in enumerate(RBM.all_configs(self.N, LBS=LBS)):
            eh = self.expectH(c)
            unpv = exp(-self.fe(c)) #unnormalized p(v)
            dWneg += dot(eh.T*unpv, c)
            daneg += (eh*unpv[:, newaxis]).sum(axis=0)
            dbneg += (c*unpv[:, newaxis]).sum(axis=0)
            logZ[i] = logSumExp(-self.fe(c))
            dfe(c, dfe_tmp)
            dfecov += dot(dfe_tmp.T*unpv, dfe_tmp)
            dfemean+= dot(unpv, dfe_tmp)
        logZ = logSumExp(logZ)
        dWneg /= exp(logZ)
        daneg /= exp(logZ)
        dbneg /= exp(logZ)
        dfecov /= exp(logZ)
        dfemean/= exp(logZ)
        grad = npcat(((dWpos-dWneg).reshape(self.M*self.N),dapos-daneg, dbpos-dbneg), axis=0)
        G = dfecov-dfemean[:, newaxis]*(dfemean)
        u,l,v = np.linalg.svd(G)
        natgrad = (v.T/l).dot(u.T.dot(grad))
        #print natgrad.shape
        #print grad.shape
        natgrad[:self.M*self.N].reshape(self.M, self.N)
        natgrad[self.M*self.N:self.M*self.N+self.M].reshape(self.M)
        natgrad[self.M*self.N+self.M:].reshape(self.N)
        return (dWpos, dapos, dbpos), (dWneg, daneg, dbneg), logZ, grad, natgrad

    def AIS(self, betaseq, N=100, base_params=None, mode='logZ', verbose=False, data=None):
        if data is None:
            print 'Collecting target samples...'
            data = self.gen_train_samples(size=50,nbatches=1000)
            print 'Done'
        if not(base_params):
            print 'Computing the initial parameters...'
            pm = data.mean(axis=0)
            pm[pm<0.01] = 0.01
            base_params = log(pm) - log(1-pm)
            print 'Done'

        bA = base_params
        MA = self.shape[0]
        WB, aB, bB = self.W, self.a, self.b
        logZA =MA*log(2) + ReL(bA).sum()
        prob = sigmoid(tile(bA, [N, 1]))
        vis = prob > rand(*prob.shape)
        logw = -(vis.dot(bA) + MA*log(2))
        logw_ = zeros(logw.shape)
        for i,beta in enumerate(betaseq[1:-1]):
            if verbose and (i%100==0):
                print 'i:%g'%i
                print 'ESS:%g/%g'%(exp(2*logSumExp(logw_)-logSumExp(2*logw_)),N)
            logw += (1-beta)*vis.dot(bA) + beta*vis.dot(bB) + ReL(beta*(vis.dot(WB.T)+aB)).sum(axis=1)
            logw_[:] = logw
            prob = sigmoid(beta*(vis.dot(WB.T)+aB))
            hid  = prob > rand(*prob.shape)
            prob = sigmoid((1-beta)*bA + beta*(hid.dot(WB)+bB))
            vis  = prob > rand(*prob.shape)
            logw -= (1-beta)*vis.dot(bA) + beta*vis.dot(bB) + ReL(beta*(vis.dot(WB.T)+aB)).sum(axis=1)
        logw += vis.dot(bB) + ReL((vis.dot(WB.T)+aB)).sum(axis=1)
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

    def sample(self, T=100, nbatches=100, v0=None):
        if v0 is None:
            v0 = random.rand(nbatches, self.shape[1])>0.5
        return super(BRBM, self).sample(T=T,nbatches=nbatches, v0=v0)


    def gen_train_samples(self, size=100, freq=100, burnin=100, nbatches=100):
        'generate samples for training'
        vtmp = random.rand(nbatches, self.shape[1])>0.5
        v = zeros((size, nbatches, self.shape[1]))
        #burn in
        for t in xrange(burnin):
            vtmp = self.sampleV(self.sampleH(vtmp)[0])[0]
        #sample
        for i in xrange(size):
            for t in xrange(freq):
                vtmp = self.sampleV(self.sampleH(vtmp)[0])[0]
            v[i] = vtmp
        return v.reshape(nbatches*size, self.shape[1])

    def AIS_NGGA(self, learning_rates, betaseq, N=100, base_rbm=None, data=None, mode='logZ'):
        if data is None:
            data = rbm.gen_train_samples()
            data = data[random.permutation(data.shape[0])]
        if base_rbm is None:
            base_rbm = basemodel_for(data, batchsz=100)
        rbm_params,logZA, logw, vis = self.AIS_NG(learning_rates, N, base_rbm, data, mode='preprocess')
        logw = self.AIS_GA(betaseq,  base_params=rbm_params, logw=logw, vis=vis)

        r_AIS = logSumExp(logw) - log(N)
        logZB = r_AIS + logZA 

        meanlogw = logw.mean()
        logstd_AIS = log(std(exp(logw-meanlogw))) + meanlogw -log(N)/2
        logZB_est_bounds = (logDiffExp(asarray([log(3)+logstd_AIS, r_AIS]))[0]+logZA,
                            logSumExp( asarray([log(3)+logstd_AIS, r_AIS]))   +logZA)


    def AIS_GA(self, betaseq, base_params, logw, vis):
        N = len(logw)
        WA, aA, bA = base_params
        M = base_rbm.shape[0]
        WB, aB, bB = self.W, self.a, self.b

        logw -= vis.dot(bA) + ReL((vis.dot(WA.T)+aA)).sum(axis=1)
        for beta in betaseq[1:-1]:
            logw += ((1-beta)*vis.dot(bA) + ReL((1-beta)*(vis.dot(WA.T)+aA)).sum(axis=1) + 
                      beta*vis.dot(bB)    + ReL(beta*(vis.dot(WB.T)+aB)).sum(axis=1))
            probA = sigmoid((1-beta)*(vis.dot(WA.T)+aA))
            hidA  = probA > rand(*probA.shape)
            probB = sigmoid(beta*(vis.dot(WB.T)+aB))
            hidB  = probB > rand(*probB.shape)
            prob = sigmoid((1-beta)*(hidA.dot(WA)+bA) + beta*(hidB.dot(WB)+bB))
            vis  = prob > rand(*prob.shape)
            logw -= ((1-beta)*vis.dot(bA) + ReL((1-beta)*(vis.dot(WA.T)+aA)).sum(axis=1) + 
                      beta*vis.dot(bB)    + ReL(beta*(vis.dot(WB.T)+aB)).sum(axis=1))
        logw += vis.dot(bB) + ReL((vis.dot(WB.T)+aB)).sum(axis=1)

        return logw

    def AIS_NG(self, learning_rates, N=100, base_rbm=None, data=None, mode='logZ', threshold=0.03, verbose=True):
        'AIS with natural gradient'
        if data is None:
            print 'Collecting target samples...'
            data = self.gen_train_samples(size=50,nbatches=1000)
            print 'Done'
        if not(base_rbm):
            print 'Computing the initial parameters...'
            #base_rbm = BRBM(M=self.M, N=self.N)
            pm = data.mean(axis=0)
            pm[pm<0.01] = 0.01
            bA = log(pm) - log(1-pm)
            print 'Done'
        nbatches=data.shape[0]/self.batchsz
        data = data.reshape(nbatches, self.batchsz, data.shape[1])

        #base
        MA = self.M
        logZA =MA*log(2) + ReL(bA).sum()
        #sample from pA
        prob = sigmoid(tile(bA, [N, 1]))
        vis = prob > rand(*prob.shape)
        #visold = prob > rand(*prob.shape)
        prob = sigmoid(tile(bA, [self.batchsz, 1]))
        vis_PCD = prob > rand(*prob.shape)
        #init AIS weights
        logw = -(vis.dot(bA) + MA*log(2))
        r_AIS = 0.0

        #init params
        #W = zeros(self.W.shape, dtype=float)
        W = 0.01*random.randn(*self.W.shape)
        a = zeros(self.a.shape, dtype=float)
        b = bA

        hnum= self.M
        vnum= self.N
        vhnum    = vnum*hnum
        paramnum = vnum*hnum + vnum + hnum

        def dfe(vis):
            dW = -sigmoid(vis.dot(W.T)+a).reshape(N, hnum, 1) * vis.reshape(N, 1, vnum)
            da = -sigmoid(vis.dot(W.T)+a)
            db = -vis
            return dW.reshape(N,vhnum), da, db

        #init Riemannian square distance
        sqdist=0.
        sqdist_tmp=0.

        #initial AIS weights
        weight= ones((N, 1),dtype=float)
        norm  = weight.sum()

        #init fisher metric
        X = zeros((N+1, paramnum), dtype=float)
        Dinv = zeros(N+1, dtype=float)
        X, Dinv= update_X(dfe(vis), X, Dinv, weight)

        do_PCD = False

        if verbose:
            print 'Starting AIS runs...'

        for i,lr in enumerate(learning_rates[:-1]):
            if verbose and (i%max(5,len(learning_rates)/100) == 0):
                print 'i:%g'%i
                print 'square Riemannian dist:%g'%(sqdist+sqdist_tmp)
                print 'ESS:%g/%g'%((weight.sum()**2)/(weight**2).sum(), N)
                NLL = -(asarray([d.dot(b) + ReL(d.dot(W.T)+a).sum(axis=1) for d in data]).sum()/(nbatches*self.batchsz) - (r_AIS + logZA))
                print 'NLL:%g'%NLL
                ERR = BRBM.compute_err_of((W,a,b), data)
                print 'ERR:%g'%ERR
                plt.subplot(2,1,1)
                plt.imshow(tile_raster_images(vis, (28,28), (int(ceil(float(N)/50)),50)), cmap=cm.gray)
                plt.subplot(2,1,2)
                plt.imshow(tile_raster_images(W, (28,28), (int(ceil(float(hnum)/50)),50)), cmap=cm.gray)
                plt.draw();plt.draw()
                
            #param update by natural gradient
            dW = sigmoid(data[i%nbatches].dot(W.T)+a).T.dot(data[i%nbatches])/self.batchsz 
            #dW-= (sigmoid(visold.dot(W.T)+a)*weight).T.dot(visold)/norm
            dW-= sigmoid(vis_PCD.dot(W.T)+a).T.dot(vis_PCD)/self.batchsz 
            da = sigmoid(data[i%nbatches].dot(W.T)+a).mean(axis=0) - sigmoid(vis_PCD.dot(W.T)+a).mean(axis=0)
            db = data[i%nbatches].mean(axis=0) - vis_PCD.mean(axis=0)
            if not(do_PCD):
                #print 'nat'
                natgrad, dmetric = compute_NG((dW, da, db), self.shape, X, Dinv, do_PCD)
                dW_ng, da_ng, db_ng = natgrad
                W += lr*dW_ng
                a += lr*da_ng
                b += lr*db_ng
                sqdist_tmp += (lr**2)*dmetric
            else:
                #print 'PCD'
                W += lr*dW
                a += lr*da
                b += lr*db

            #pos AIS weight update
            logw += vis.dot(b) + ReL(vis.dot(W.T)+a).sum(axis=1)
            weight= np.exp(logw)[:, newaxis]
            #if do_PCD:
            #    weight = ones(logw.shape)[:, newaxis]
            #norm  = weight.sum()
            #visold = vis.copy()##

            #update G
            if sqdist_tmp > threshold:
                sqdist += sqdist_tmp
                sqdist_tmp = 0.
                X, Dinv= update_X(dfe(vis), X, Dinv, weight)

            #sample
            prob = sigmoid(vis.dot(W.T)+a)
            hid  = prob > rand(*prob.shape)
            prob = sigmoid(hid.dot(W)+b)
            vis  = prob > rand(*prob.shape)
            #neg AIS weight update
            logw -= vis.dot(b) + ReL(vis.dot(W.T)+a).sum(axis=1)
            #sample for PCD
            prob = sigmoid(vis_PCD.dot(W.T)+a)
            hid  = prob > rand(*prob.shape)
            prob = sigmoid(hid.dot(W)+b)
            vis_PCD  = prob > rand(*prob.shape)

        #param update by natural gradient
        i+=1
        lr=learning_rates[-1]
        dW = sigmoid(data[i%nbatches].dot(W.T)+a).T.dot(data[i%nbatches])/self.batchsz 
        #dW-= (sigmoid(vis.dot(W.T)+a)*weight).T.dot(vis)/norm
        dW-= sigmoid(vis_PCD.dot(W.T)+a).T.dot(vis_PCD)/self.batchsz 
        #da = sigmoid(data[i%nbatches].dot(W.T)+a).mean(axis=0) - (sigmoid(vis.dot(W.T)+a)*weight).sum(axis=0)/norm
        #db = data[i%nbatches].mean(axis=0) - (vis*weight).sum(axis=0)/norm
        da = sigmoid(data[i%nbatches].dot(W.T)+a).mean(axis=0) - sigmoid(vis_PCD.dot(W.T)+a).mean(axis=0)
        db = data[i%nbatches].mean(axis=0) - vis_PCD.mean(axis=0)
        if not(do_PCD):
            #print 'nat'
            natgrad, dmetric = compute_NG((dW, da, db), self.shape, X, Dinv, do_PCD)
            dW_ng, da_ng, db_ng = natgrad
            W += lr*dW_ng
            a += lr*da_ng
            b += lr*db_ng
            sqdist_tmp += (lr**2)*dmetric
        else:
            #print 'PCD'
            W += lr*dW
            a += lr*da
            b += lr*db

        #finalize AIS weights
        logw += vis.dot(b) + ReL(vis.dot(W.T)+a).sum(axis=1)

        r_AIS = logSumExp(logw) - log(N)
        logZB = r_AIS + logZA 

        meanlogw = logw.mean()
        logstd_AIS = log(std(exp(logw-meanlogw))) + meanlogw -log(N)/2
        logZB_est_bounds = (logDiffExp(asarray([log(3)+logstd_AIS, r_AIS]))[0]+logZA,
                            logSumExp( asarray([log(3)+logstd_AIS, r_AIS]))   +logZA)
        if mode == 'logZ':
            return logZB, logZB_est_bounds, (W,a,b)
        elif mode == 'preprocess':
            return (W,a,b), logZA, logw, vis
        else:
            return (hid, vis), logZB, logZB_est_bounds


    @staticmethod
    def compute_err_of(params, data):
        W, a, b = params
        err = 0
        for d in data:
            prob = sigmoid(d.dot(W.T)+a)
            hid  = prob > rand(*prob.shape)
            prob = sigmoid(hid.dot(W)+b)
            err += -((d*log(prob)+(1-d)*log(1-prob)).sum(axis=1)).mean()
        return err/len(data)

def test_AIS_NG():
    rbm = BRBM(M=10, N=10)
    rbm.W *= 10
    rbm.AIS_NG(0.01*ones(1000), N=1000)


def test_AIS_NG2():
    import DBNs.rbm
    from Basics.utils import pickle
    plt.ion()
    plt.figure()
    #rbmdir= '../DBNs/CD25_20/rbm_lr0.06_dr1e-05'
    rbmdir= '../DBNs/CD25_500/rbm_lr0.1_dr0_mom0'
    rbm = DBNs.rbm.load_brbm(rbmdir)
    data = DBNs.rbm.gen_train(rbmdir)
    params, logZA, logw, vis = rbm.AIS_NG(lrseq([0.1, 0.05], [5000, 5000]), N=500, data=data, mode='preprocess')
    pickle('foo.pkl.gz', (params, logZA, logw, vis))


def test_AIS_NG3():
    import DBNs.rbm
    from Basics.utils import pickle, unpickle
    plt.ion()
    plt.figure()
    #rbmdir= '../DBNs/CD25_20/rbm_lr0.06_dr1e-05'
    rbmdir= '../DBNs/CD25_500/rbm_lr0.1_dr0_mom0'
    rbm = DBNs.rbm.load_brbm(rbmdir)
    #data = DBNs.rbm.gen_train(rbmdir)
    data = unpickle('../data/mnist.pkl.gz')[0][0]
    params, logZA, logw, vis = rbm.AIS_NG(lrseq([0.001, 0.0005, 0.000025], [20000, 10000, 10000]), N=500, data=data, mode='preprocess')
    pickle('foo.pkl.gz', (params, logZA, logw, vis))

def lrseq(lrs, nums):
    return npcat([lr*ones(num) for lr,num in zip(lrs,nums)], axis=0)

def update_X(dfe, X, Dinv, weight):
    dW_, da_, db_ = dfe
    vnum = db_.shape[1]
    hnum = da_.shape[1]
    vhnum=vnum*hnum
    nparams = vhnum+hnum+vnum
    N = dW_.shape[0]
    if weight is None:
        weight = ones(N, dtype=float)
        weight = weight[:, newaxis]
    norm = weight.sum()
    #
    dWm_ = (dW_*weight).sum(axis=0)/norm
    dam_ = (da_*weight).sum(axis=0)/norm
    dbm_ = (db_*weight).sum(axis=0)/norm

    X[:-1, :vhnum] = dW_
    X[:-1, vhnum:vhnum+hnum] = da_
    X[:-1, vhnum+hnum:] = db_
    X[-1, :vhnum] = dWm_
    X[-1, vhnum:vhnum+hnum] = dam_
    X[-1, vhnum+hnum:] = dbm_
    Dinv[arange(N)] = weight.flat/norm
    Dinv[-1] = -1.
    return X,Dinv

def compute_NG(dparams, shape, X, Dinv, flag=True):
    dW, da, db = dparams
    M,N = shape
    dparams = zeros(M*N+M+N, dtype=float)
    dparams[:M*N] = dW.flat
    dparams[M*N:M*N+M] = da
    dparams[M*N+M:] = db

    #compute inverse of (lI+G)
    l = 10.
    #Ginv = - X.T.dot(linalg.inv(diag(1/Dinv) + X.dot(X.T)/l)).dot(X)/(l**2)
    #Ginv[arange(M*N+M+N), arange(M*N+M+N)] += 1/l 
    #print 'Ginv'
    #print Ginv
    if flag:
        dparams_nat = dparams/l - X.T.dot(linalg.inv(diag(1/Dinv) + X.dot(X.T)/l).dot(X.dot(dparams)))/(l**2)
    else:
        dparams_nat = dparams
    dmetric = dparams_nat.dot(dparams)

    natgrad = (dparams_nat[:M*N].reshape(M,N),
               dparams_nat[M*N:M*N+M],
               dparams_nat[M*N+M:])
    return natgrad, dmetric


@static_var('Gtmp', None)
def fisher_inv_naive(dfe, weight=None):
    Gtmp=fisher_inv_naive.Gtmp
    dW_, da_, db_ = dfe
    vnum = db_.shape[1]
    hnum = da_.shape[1]
    vhnum=vnum*hnum
    if weight is None:
        weight = ones(dW_.shape[0], dtype=float)
        weight = weight[:, newaxis]
    norm = weight.sum()
    #
    dWm_ = (dW_*weight).sum(axis=0)/norm
    dam_ = (da_*weight).sum(axis=0)/norm
    dbm_ = (db_*weight).sum(axis=0)/norm
    #diagonal submatrices
    Gtmp[:vhnum, :vhnum]                     = (dW_*weight).T.dot(dW_)/norm - dWm_[:, newaxis]*dWm_
    Gtmp[vhnum:vhnum+hnum, vhnum:vhnum+hnum] = (da_*weight).T.dot(da_)/norm - dam_[:, newaxis]*dam_
    Gtmp[vhnum+hnum:, vhnum+hnum:]           = (db_*weight).T.dot(db_)/norm - dbm_[:, newaxis]*dbm_
    #upper-triangle submatrices
    Gtmp[:vhnum, vhnum:vhnum+hnum]      = (dW_*weight).T.dot(da_)/norm - dWm_[:, newaxis]*dam_
    Gtmp[:vhnum, vhnum+hnum:]           = (dW_*weight).T.dot(db_)/norm - dWm_[:, newaxis]*dbm_
    Gtmp[vhnum:vhnum+hnum, vhnum+hnum:] = (da_*weight).T.dot(db_)/norm - dam_[:, newaxis]*dbm_
    #lower-triangle submatrices
    Gtmp[vhnum:vhnum+hnum,:vhnum]       = Gtmp[:vhnum, vhnum:vhnum+hnum].T
    Gtmp[vhnum+hnum:, :vhnum]           = Gtmp[:vhnum, vhnum+hnum:].T
    Gtmp[vhnum+hnum:, vhnum:vhnum+hnum] = Gtmp[vhnum:vhnum+hnum, vhnum+hnum:].T
    #compute inverse
    l = 1.
    Ginv = np.linalg.inv(l*eye(Gtmp.shape[0]) + Gtmp)
    print 'fisher_inv_naive'
    print 'G'
    print Gtmp
    print 'Ginv'
    print Ginv
    #A = Gtmp+0.01*eye(Gtmp.shape[0])
    #Ginv = approx_matrix_inv(A, V_0=eye(Gtmp.shape[0])/(np.sqrt(np.sum(A**2))), niter=50)

    #return as tensors
    Ginv = (Ginv[:vhnum, :vhnum].reshape(hnum,vnum,hnum,vnum), 
            Ginv[:vhnum, vhnum:vhnum+hnum].reshape(hnum, vnum, hnum),
            Ginv[:vhnum, vhnum+hnum:].reshape(hnum, vnum, vnum),
            Ginv[vhnum:vhnum+hnum, vhnum:vhnum+hnum], 
            Ginv[vhnum:vhnum+hnum, vhnum+hnum:], 
            Ginv[vhnum+hnum:, vhnum+hnum:])
    return Ginv


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
        assert(self.M < 25)
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
        for i,c in enumerate(RBM.all_configs(self.M, LBS=LBS)):
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
        for i,c in enumerate(RBM.all_configs(self.M, LBS=LBS)):
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


    def gen_train_samples(self, size=100, freq=100, burnin=100, nbatches=100):
        'generate samples for training'
        vtmp = random.randn(nbatches, self.shape[1])*10
        v = zeros((size, nbatches, self.shape[1]))
        #burn in
        for t in xrange(burnin):
            vtmp = self.sampleV(self.sampleH(vtmp)[0])[0]
        #sample
        for i in xrange(size):
            for t in xrange(freq):
                vtmp = self.sampleV(self.sampleH(vtmp)[0])[0]
            v[i] = vtmp
        return v.reshape(nbatches*size, self.shape[1])

    def AIS_NG(self, learning_rates, N=100, base_params=None, data=None, mode='logZ', threshold=0.1,
               debug = False, verbose=True):
        'AIS with natural gradient'

        if base_params is None:
            print 'Computing the initial parameters...'
            base_params = self.cov(1.0)
            print 'Done'
        if data is None:
            print 'Collecting target samples...'
            data = self.gen_train_samples(size=50,nbatches=1000)
            print 'Done'
        nbatches=data.shape[0]/self.batchsz
        data = data.reshape(nbatches, self.batchsz, data.shape[1])

        #base 
        MA = self.M
        bA, sA = base_params[0], np.sqrt(diag(base_params[1]))
        logZA =MA*log(2) + 0.5*self.N*log(2*np.pi) + log(sA).sum()

        #sample from pA
        prob = tile(bA, [N, 1])
        vis = randn(*prob.shape)*sA + prob
        visold = randn(*prob.shape)*sA + prob

        #init AIS weights
        logw =  0.5*(((vis-bA)/sA)**2).sum(axis=1) - MA*log(2)
        r_AIS = 0.0

        #init params
        W = zeros(self.W.shape, dtype=float)
        a = zeros(self.a.shape, dtype=float)
        b = bA
        s = sA

        hnum= self.M
        vnum= self.N
        vhnum    = vnum*hnum
        paramnum = vnum*hnum + 2*vnum + hnum

        def dfe(vis):
            dW = -sigmoid((vis/s).dot(W.T)+a).reshape(N, hnum, 1) * (vis/s).reshape(N, 1, vnum)
            da = -sigmoid((vis/s).dot(W.T)+a)
            db = -(vis-b)/(s**2)
            ds = -((vis-b)**2)/(s**3)
            return dW.reshape(N,vhnum), da, db, ds

        #init Riemannian square distance
        sqdist=0.
        sqdist_tmp=0.

        #initial AIS weights
        weight= ones((N, 1),dtype=float)
        norm  = weight.sum()

        #init fisher metric
        fisher_inv_GRBM.Gtmp = zeros((paramnum, paramnum))
        Ginv,G = fisher_inv_GRBM(dfe(vis), weight)
        
        if debug:
            assert(self.N==2)
            ext = asarray([[-8,8],[-8,8]])
            X, Y = np.meshgrid(*mesh(ext))
            XY = npcat((X[:, :, newaxis], Y[:, :, newaxis]), axis=2)
        if verbose:
            print 'Starting AIS runs...'

        for i,lr in enumerate(learning_rates[:-1]):
            if verbose and (i%(len(learning_rates)/100) == 0):
                print 'i:%g'%i
                print 'square Riemannian dist:%g'%(sqdist+sqdist_tmp)
                print 'ESS:%g/%g'%((weight.sum()**2)/(weight**2).sum(), N)
                NLL = -(asarray([-0.5*(((d-b)/s)**2).sum(axis=1) + ReL((d/s).dot(W.T)+a).sum(axis=1) for d in data]).sum()/(nbatches*self.batchsz) - (r_AIS + logZA))
                print 'NLL:%g'%NLL
                if debug:
                    print W
            if debug and (i%(len(learning_rates)/5) == 0):
                fmap = np.exp(-0.5*(((XY-b)/s)**2).sum(axis=2) + ReL(np.tensordot((XY/s), W.T, axes=[2,0])+a).sum(axis=2))
                fig = plt.figure()
                ax = fig.add_subplot(111, aspect='equal')
                plt.imshow(fmap, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=ext.reshape(4))
                plt.scatter(vis[:,0], vis[:,1], c='cyan', s=30)
                ax.set_xlim(*ext[0])
                ax.set_ylim(*ext[1])
                ax.set_axis_off()
                plt.show()
            #param update by natural gradient
            eh_d = sigmoid((data[i%nbatches]/s).dot(W.T)+a)
            eh_m = sigmoid((visold/s).dot(W.T)+a)
            dW = eh_d.T.dot(data[i%nbatches]/s)/self.batchsz 
            dW-= (eh_m*weight).T.dot(visold/s)/norm
            da = eh_d.mean(axis=0) 
            da-= (eh_m*weight).sum(axis=0)/norm
            db = np.mean((data[i%nbatches]-b)/(s**2), axis=0) 
            db-= np.sum(((visold-b)/(s**2))*weight, axis=0)/norm
            ds = np.mean(((data[i%nbatches]-b)**2)/(s**3), axis=0) 
            ds-= np.sum((((visold-b)**2)/(s**3))*weight, axis=0)/norm
            dW_ng = lr*(np.tensordot(Ginv[0], dW, axes=([2,3], [0,1])) + 
                        np.tensordot(Ginv[1], da, axes=([2], [0])) + 
                        np.tensordot(Ginv[2], db, axes=([2], [0])) + 
                        np.tensordot(Ginv[3], ds, axes=([2], [0])))
            da_ng = lr*(np.tensordot(Ginv[1], dW, axes=([0,1], [0,1])) + 
                        np.dot(Ginv[4], da) +
                        np.dot(Ginv[5], db) + 
                        np.dot(Ginv[6], ds))
            db_ng = lr*(np.tensordot(Ginv[2], dW, axes=([0,1], [0,1])) +
                        np.dot(Ginv[5].T, da) +
                        np.dot(Ginv[7], db) + 
                        np.dot(Ginv[8], ds))
            ds_ng = lr*(np.tensordot(Ginv[3], dW, axes=([0,1], [0,1])) +
                        np.dot(Ginv[6].T, da) +
                        np.dot(Ginv[8].T, db) + 
                        np.dot(Ginv[9], ds))
            W += dW_ng
            a += da_ng
            b += db_ng
            s += ds_ng
            sqdist_tmp += (dW_ng*dW).sum() + (da_ng*da).sum() + (db_ng*db).sum() + (ds_ng*ds).sum()

            #pos AIS weight update
            logw += -0.5*(((vis-b)/s)**2).sum(axis=1) + ReL((vis/s).dot(W.T)+a).sum(axis=1)
            weight= np.exp(logw)[:, newaxis]
            norm  = weight.sum()
            visold = vis.copy()##
            r_AIS = logSumExp(logw) - log(N)

            #update G
            if sqdist_tmp > threshold:
                sqdist += sqdist_tmp
                sqdist_tmp = 0.
                Ginv,G = fisher_inv_GRBM(dfe(vis), weight)

            #sample
            prob = sigmoid((vis/s).dot(W.T)+a)
            hid  = prob > rand(*prob.shape)
            prob = s*np.dot(hid, W)+b
            vis = randn(*prob.shape)*s + prob
            #neg AIS weight update
            logw -= -0.5*(((vis-b)/s)**2).sum(axis=1) + ReL((vis/s).dot(W.T)+a).sum(axis=1)

        #param update by natural gradient
        i+=1
        lr=learning_rates[-1]

        eh_d = sigmoid((data[i%nbatches]/s).dot(W.T)+a)
        eh_m = sigmoid((visold/s).dot(W.T)+a)
        dW = eh_d.T.dot(data[i%nbatches]/s)/self.batchsz 
        dW-= (eh_m*weight).T.dot(visold/s)/norm
        da = eh_d.mean(axis=0) 
        da-= (eh_m*weight).sum(axis=0)/norm
        db = np.mean((data[i%nbatches]-b)/(s**2), axis=0) 
        db-= np.sum(((visold-b)/(s**2))*weight, axis=0)/norm
        ds = np.mean(((data[i%nbatches]-b)**2)/(s**3), axis=0) 
        ds-= np.sum((((visold-b)**2)/(s**3))*weight, axis=0)/norm
        dW_ng = lr*(np.tensordot(Ginv[0], dW, axes=([2,3], [0,1])) + 
                    np.tensordot(Ginv[1], da, axes=([2], [0])) + 
                    np.tensordot(Ginv[2], db, axes=([2], [0])) + 
                    np.tensordot(Ginv[3], ds, axes=([2], [0])))
        da_ng = lr*(np.tensordot(Ginv[1], dW, axes=([0,1], [0,1])) + 
                    np.dot(Ginv[4], da) +
                    np.dot(Ginv[5], db) + 
                    np.dot(Ginv[6], ds))
        db_ng = lr*(np.tensordot(Ginv[2], dW, axes=([0,1], [0,1])) +
                    np.dot(Ginv[5].T, da) +
                    np.dot(Ginv[7], db) + 
                    np.dot(Ginv[8], ds))
        ds_ng = lr*(np.tensordot(Ginv[3], dW, axes=([0,1], [0,1])) +
                    np.dot(Ginv[6].T, da) +
                    np.dot(Ginv[8].T, db) + 
                    np.dot(Ginv[9], ds))
        W += dW_ng
        a += da_ng
        b += db_ng
        s += ds_ng
        sqdist_tmp += (dW_ng*dW).sum() + (da_ng*da).sum() + (db_ng*db).sum() + (ds_ng*ds).sum()


        #finalize AIS weights
        logw += -0.5*(((vis-b)/s)**2).sum(axis=1) + ReL((vis/s).dot(W.T)+a).sum(axis=1)

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
            return (hid, vis), logZB, logZB_est_bounds, ESS

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


    def cov_opt(self, base_params=None, nbatches=100, T=5000, niter=20):
        '''
        A routine for optimal covariance matrix.
        This function numerically approximates the optimal covariance matrix
        for an initial distribition that minimizes the Hellinger
        distance between the initial and target distributions. 
        This routine relies on two approximation techniques: 
        the fixed point method (Minka, 2006) and approximated 
        geometric path (Salakhutdinov and Murray, 2008).        
        '''
        if base_params is None:
            base_params = self.cov(1.1)
        for i in xrange(niter):
            v, x, h = self.sample_geometric_avr_cov(self, beta=0.5, base_params=base_params, nbatches=nbatches, T=T)
            v = v[100:].reshape((T-100)*nbatches, self.N)
            base_params = v.mean(axis=0), scale*cov(v.T)
        return base_params

    def measure_hellinger(self, params, nsamples):
        '''
        Compute an MCMC estimate of 
        $\sqrt(Z^B)\left(\left\{\frac{H(p,q)}{2}\right\}^2-1\right)$
        that is monotonic to the Hellinger distance $H(p,q)$
        '''
        mean, cov = params
        covinv = linalg.inv(cov)
        detcov = linalg.det(cov)
        def logpdf_norm(x):
            return - 0.5*(((x-mean).dot(covinv))*(x-mean)).sum(axis=1) - 0.5*self.N*log(2*np.pi) - 0.5*log(detcov)
        x = random.multivariate_normal(mean, cov, nsamples)
        tmp = (exp(-0.5*self.fe(x) - 0.5*logpdf_norm(x)))
        return -tmp.mean(), tmp.std()

    def sample_geometric_avr_cov(self, beta=0.5, base_params=None, nbatches=100, T=1000):
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


@static_var('Gtmp', None)
def fisher_inv_GRBM(dfe, weight=None):
    dW_, da_, db_, ds_ = dfe
    Gtmp=fisher_inv_GRBM.Gtmp
    vnum = db_.shape[1]
    hnum = da_.shape[1]
    vhnum=vnum*hnum
    if weight is None:
        weight = ones(dW_.shape[0], dtype=float)
        weight = weight[:, newaxis]
    norm = weight.sum()
    #
    dWm_ = (dW_*weight).sum(axis=0)/norm
    dam_ = (da_*weight).sum(axis=0)/norm
    dbm_ = (db_*weight).sum(axis=0)/norm
    dsm_ = (ds_*weight).sum(axis=0)/norm
    #diagonal submatrices
    Gtmp[:vhnum, :vhnum]               = (dW_*weight).T.dot(dW_)/norm - dWm_[:, newaxis]*dWm_
    Gtmp[vhnum:vhnum+hnum, 
         vhnum:vhnum+hnum]             = (da_*weight).T.dot(da_)/norm - dam_[:, newaxis]*dam_
    Gtmp[vhnum+hnum:vhnum+hnum+vnum,
         vhnum+hnum:vhnum+hnum+vnum]   = (db_*weight).T.dot(db_)/norm - dbm_[:, newaxis]*dbm_
    Gtmp[vhnum+hnum+vnum:, 
         vhnum+hnum+vnum:]             = (ds_*weight).T.dot(ds_)/norm - dsm_[:, newaxis]*dsm_
    #upper-triangle submatrices
    #
    Gtmp[:vhnum, vhnum:vhnum+hnum]          = (dW_*weight).T.dot(da_)/norm - dWm_[:, newaxis]*dam_
    Gtmp[:vhnum, vhnum+hnum:vhnum+hnum+vnum]= (dW_*weight).T.dot(db_)/norm - dWm_[:, newaxis]*dbm_
    Gtmp[:vhnum, vhnum+hnum+vnum:]          = (dW_*weight).T.dot(ds_)/norm - dWm_[:, newaxis]*dsm_
    #
    Gtmp[vhnum:vhnum+hnum, 
         vhnum+hnum:vhnum+hnum+vnum]    = (da_*weight).T.dot(db_)/norm - dam_[:, newaxis]*dbm_
    Gtmp[vhnum:vhnum+hnum, 
         vhnum+hnum+vnum:]              = (da_*weight).T.dot(ds_)/norm - dam_[:, newaxis]*dsm_
    #
    Gtmp[vhnum+hnum:vhnum+hnum+vnum, 
         vhnum+hnum+vnum:]              = (db_*weight).T.dot(ds_)/norm - dbm_[:, newaxis]*dsm_

    #lower-triangle submatrices
    Gtmp[vhnum:vhnum+hnum,:vhnum]       = Gtmp[:vhnum, vhnum:vhnum+hnum].T
    Gtmp[vhnum+hnum:vhnum+hnum+vnum,
         :vhnum]                        = Gtmp[:vhnum, vhnum+hnum:vhnum+hnum+vnum].T
    Gtmp[vhnum+hnum+vnum:, :vhnum]      = Gtmp[:vhnum, vhnum+hnum+vnum:].T
    #
    Gtmp[vhnum+hnum:vhnum+hnum+vnum,
         vhnum:vhnum+hnum]              = Gtmp[vhnum:vhnum+hnum, vhnum+hnum:vhnum+hnum+vnum].T
    Gtmp[vhnum+hnum+vnum:,
         vhnum:vhnum+hnum]              = Gtmp[vhnum:vhnum+hnum, vhnum+hnum+vnum:].T
    #
    Gtmp[vhnum+hnum+vnum:,
         vhnum+hnum:vhnum+hnum+vnum]    = Gtmp[vhnum+hnum:vhnum+hnum+vnum, vhnum+hnum+vnum:].T

    #compute inverse
    Ginv = np.linalg.inv(Gtmp+0.01*eye(Gtmp.shape[0]))
    #return as tensors
    Ginv = (Ginv[:vhnum, :vhnum].reshape(hnum,vnum,hnum,vnum), 
            Ginv[:vhnum, vhnum:vhnum+hnum].reshape(hnum, vnum, hnum),
            Ginv[:vhnum, vhnum+hnum:vhnum+hnum+vnum].reshape(hnum, vnum, vnum),
            Ginv[:vhnum, vhnum+hnum+vnum:].reshape(hnum, vnum, vnum),
            #
            Ginv[vhnum:vhnum+hnum, vhnum:vhnum+hnum], 
            Ginv[vhnum:vhnum+hnum, vhnum+hnum:vhnum+hnum+vnum], 
            Ginv[vhnum:vhnum+hnum, vhnum+hnum+vnum:], 
            #
            Ginv[vhnum+hnum:vhnum+hnum+vnum, vhnum+hnum:vhnum+hnum+vnum],
            Ginv[vhnum+hnum:vhnum+hnum+vnum, vhnum+hnum+vnum:],
            #
            Ginv[vhnum+hnum+vnum:, vhnum+hnum+vnum:])

    G    = (Gtmp[:vhnum, :vhnum].reshape(hnum,vnum,hnum,vnum), 
            Gtmp[:vhnum, vhnum:vhnum+hnum].reshape(hnum, vnum, hnum),
            Gtmp[:vhnum, vhnum+hnum:vhnum+hnum+vnum].reshape(hnum, vnum, vnum),
            Gtmp[:vhnum, vhnum+hnum+vnum:].reshape(hnum, vnum, vnum),
            #
            Gtmp[vhnum:vhnum+hnum, vhnum:vhnum+hnum], 
            Gtmp[vhnum:vhnum+hnum, vhnum+hnum:vhnum+hnum+vnum], 
            Gtmp[vhnum:vhnum+hnum, vhnum+hnum+vnum:], 
            #
            Gtmp[vhnum+hnum:vhnum+hnum+vnum, vhnum+hnum:vhnum+hnum+vnum],
            Gtmp[vhnum+hnum:vhnum+hnum+vnum, vhnum+hnum+vnum:],
            #
            Gtmp[vhnum+hnum+vnum:, vhnum+hnum+vnum:])
    return Ginv, G

def base_GRBM_for(data, batchsz=100, debug=False):
    N = data.shape[1]
    rbm = BRBM(M=1, N=N)
    rbm.W[:] = 0
    rbm.a[:] = 0
    rbm.b = data.mean(axis=0)
    rbm.sigma = data.std(axis=0)
    return rbm


def load_RBM_mat(dir='M20', data=True):
    'loads GRBMs for NIPS14WS'
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

def load_BRBM_mat(dir='M20', data=True):
    import scipy.io as sio
    name = dir+'/rbm.mat'
    params = sio.loadmat(name)
    W = params['W']
    a = (params['a'].T)[0]
    b = (params['b'].T)[0]
    rbm = BRBM(M=W.shape[0], N=W.shape[1])
    rbm.W = asarray(W, dtype='>d')
    rbm.a = asarray(a, dtype='>d')
    rbm.b = asarray(b, dtype='>d')
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

