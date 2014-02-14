import __builtin__ as base
from numpy import *
import functools as ftk

import sys

import time
from pylab import *
from matplotlib.patches import Ellipse

from numpy.random import randn, rand, permutation, gamma, standard_cauchy
from numpy import linalg as LA

import MNIST
from Data import *
from rbm import *
from variedParam import * 
from grbm_M2N2 import *

from GaussianMixtures import *
from basics import *

class tinyGRBM(GRBM):
    """Tiny Gaussian RBMs"""
    def __init__(self, M, N, batchsz=100):
        super(tinyGRBM, self).__init__(M=M,N=N,batchsz=batchsz)
        self.configs = asarray(possible_configs(M), dtype=float)

    def freeEnergyOfH(self, h):
        centers = self.b + self.sigma * dot(h,self.W)
        foo = (sum((centers/self.sigma)**2, axis=1)/2)[:,newaxis] + dot(h,self.a[:,newaxis])
        return foo

    def probabilityH(self):
        fh = self.freeEnergyOfH(self.configs)
        ph = exp(fh)/sum(exp(fh))
        ph[isnan(ph)] = 1.
        return ph

    def sweepAcrossData(self,data):
        strength, target = self.sparsity.values()
        batchsz = float(self.batchsz)
        lrate, mom, drate = self.lrates()
        #main part of the training
        for item in data: ##sampled data vectors
            eh = self.expectH(item)          #expected H vectors
            dW = dot(eh.T, item/self.sigma)/batchsz
            da = mean(eh,axis=0)
            db = mean((item-self.b)/(self.sigma**2),axis=0)
            #sparsity
            da += strength*(target-da) 
            #neg. phase wo sampling
            ph = self.probabilityH()
            ev = self.expectV(self.configs)
            if any(isnan(ph)):
                break

            dW -= dot((self.configs*ph).T, ev/self.sigma)
            da -= sum(self.configs*ph,axis=0)
            db -= sum(ph*(ev-self.b)/(self.sigma**2),axis=0)
            db = zeros(db.shape)

            self.vW = mom*self.vW + lrate*dW
            self.va = mom*self.va + lrate*da
            self.vb = mom*self.vb + lrate*db
            self.W = drate*self.W + self.vW
            self.a = drate*self.a + self.va
            self.b = drate*self.b + self.vb
        print(sqrt(sum(self.W*self.W)))


def main(generator = generateData, save = {'filename':False}):
    epochs = 5000
    monitorInit()
    data = generator()
    rbm = tinyGRBM(M=4, N=2)
    rbm.algorithm = 'TRUE'
    #rbm.lrate = variedParam(0.02)
    rbm.sparsity = {'strength': .4, 'target': 0.05}
    rbm.lrate = variedParam(0.02, schedule=[['linearlyDecayFor', epochs]])
    rbm.mom   = variedParam(0.0)

    rbm.initWithData(data)
    #rbm.sigma = 0.4*rbm.sigma
    rbm.sigma = sqrt(1.0)*asarray([1.0,1.0])
    rbm.CDN = 1
    monitor(rbm, data)
    if save['filename']:
        with gzip.open(save['filename'], 'wb') as output:
            logger=genLogger(output, save['interval'])
            rbm.train(data, epochs, monitor=monitor, logger=logger)
    else:
        logger=emptyLogger
        rbm.train(data, epochs, monitor=monitor, logger=logger)

if __name__ == "__main__":
    generator, test, save = parse_command_line_args(sys.argv[1:], ['generateData12', 'False', ''])
    save = {'filename':save, 'interval':10}
    if eval(test): 
        drawData(eval(generator))
    else:
        main(eval(generator), save=save)
