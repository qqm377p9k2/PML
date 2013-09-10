import __builtin__ as base
from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

def data(N = 50, Noise=False):
    x = 2*pi*rand(N)
    y = sin(x) + 0.2*randn(N)
    if Noise:
        x[0] = 3.
        y[0] = y[0]+3.
        x[5] = 1.
        y[5] = y[5]-10.
    return (x,y)


class LSMestimator(object):
    def __init__(self, M, l=0, beta=1.):
        self.l = float(l)
        self.powers = arange(M,dtype=float)
        self.wml = None
        self.beta = beta
        self.__alpha = None

    def setAlpha(self, alpha):
        self.__alpha = alpha
        self.l = float(alpha)/self.beta

    def order(self):
        return len(self.powers)

    def fit(self, x, y):
        phi = array([power(x, p) for p in self.powers]).T
        self.wml = dot(y, dot(phi,LA.inv(self.l*eye(self.order()) + dot(phi.T,phi)).T))

    def predict(self, x):
        assert(not(self.wml is None))
        return dot(self.wml, array([power(x, p) for p in self.powers]))


def main():
    (x,y) = data()
    (xwn, ywn) = data(Noise=True)
    lsm = LSMestimator(M=6, l=2.)
    lsm2 = LSMestimator(M=6, l=2.)
    lsm.fit(x,y)
    tics = 2*pi*arange(0,1,0.02);
    lw = 5 #line width
    plt.subplot(2,1,1)
    plt.xlim(0,2*pi+2)
    plt.legend()
    plt.plot(x,y, 'ro', ms=10)
    plt.plot(tics,sin(tics), linewidth=lw, label="sin")
    plt.plot(tics,lsm.predict(tics), linewidth=lw, label="fitting");
    plt.legend(loc = 'lower right')

    ls = power(3., range(-2,3))
    print(ls)
    for idcs in range(len(ls)):
        plt.subplot(len(ls),1,idcs+1)
        plt.xlim(0,2*pi+2)
        plt.ylim(-2,2)
        plt.legend()
        plt.plot(xwn,ywn, 'ro', ms=10)
        plt.plot(tics,sin(tics), linewidth=lw, label="sin")
        lsm2.l = ls[idcs]
        lsm2.fit(xwn,ywn)
        plt.plot(tics,lsm2.predict(tics), linewidth=lw, label='l={:.1}'.format(ls[idcs]));
        plt.legend(loc = 'lower right')
    plt.show()


if __name__ == "__main__":
    main()
