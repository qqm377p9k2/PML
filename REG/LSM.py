import __builtin__ as base
from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

def data(N = 50, Noise=False, NL=0.2):
    x = 2*pi*rand(N)
    y = sin(x) + NL*randn(N)
    if Noise == True:
        Noise == 'Type1'
    if Noise == 'Type1':
        x[0] = 3.
        y[0] = y[0]+3.
        x[5] = 1.
        y[5] = y[5]-10.
    elif Noise == 'Type2':
        y[0] -= 10.
        x[0] = 5.
    return (x,y)


class LSMslv(object):
    def __init__(self, M):
        self.powers = arange(M,dtype=float)
        self.wml = None

    def order(self):
        return len(self.powers)
        
    def predict(self, x):
        assert(not(self.wml is None))
        return dot(self.wml, array([power(x, p) for p in self.powers]))


class LSM_L2(LSMslv):
    def __init__(self, M, l=0, beta=1.):
        super(self.__class__, self).__init__(M)
        self.l = float(l)
        self.beta = beta
        self.__alpha = None

    def setAlpha(self, alpha):
        self.__alpha = alpha
        self.l = float(alpha)/self.beta

    def fit(self, x, y):
        phi = array([power(x, p) for p in self.powers]).T
        self.wml = dot(y, dot(phi,LA.inv(self.l*eye(self.order()) + dot(phi.T,phi)).T))


def main():
    (x,y) = data()
    (xwn, ywn) = data(Noise=True)
    lsm = LSM_L2(M=6, l=2.)
    lsm2 = LSM_L2(M=6, l=2.)
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
