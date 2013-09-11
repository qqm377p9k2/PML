import __builtin__ as base
from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

from LSM import *

class LSM_L1(LSMslv):
    def __init__(self, M, max_steps=1e6, alpha=0.1):
        super(self.__class__, self).__init__(M)
        self.alpha = alpha
        self.wml = 0.01*randn(M)
        self.initLrate = 1e-7
        self.training_steps = max_steps

    def fit(self, x, y):
        phi = array([power(x, p) for p in self.powers])
        lrate = self.initLrate
        alpha = self.alpha
        wml = self.wml
        maxIter = int(self.training_steps/len(y))
        for i in range(maxIter):
            #lrate = self.initLrate * ((float(maxIter)/5)/(i+(float(maxIter)/5)))
            for datIdx in permutation(len(y)):
                coef = y[datIdx] - dot(wml,phi[:, datIdx])
                wml = wml + lrate*(coef*phi[:, datIdx] - alpha*sign(wml))
            err = mean((y - dot(wml,phi))**2) + alpha*sum(wml)
            if mod(i,int(500*(100./len(y))))==0:
                print(wml)
                print('Iter:' + repr(i) +'/'+ repr(maxIter) +  '\terr: ' + repr(err))
                if err < 0.01:
                    print('err is below the tolerance')
                    break
        print('Training Finished')
        self.wml = wml


def main():
    (x,y) = data(N=6, Noise=False, NL=0.3)
    lsm = LSM_L1(M=6, alpha=0.0, max_steps=5e6)

    tics = 2*pi*arange(0,1,0.02);
    lw = 5 #line width

    plt.subplot(211)
    plt.xlim(0,3*pi)
    plt.ylim(-2,2)
    plt.legend()
    plt.plot(x,y, 'ro', ms=10)
    plt.plot(tics,sin(tics), linewidth=lw, label="sin")
    lsm.fit(x,y)
    plt.plot(tics,lsm.predict(tics), linewidth=lw, label="fitting");
    plt.legend(loc = 'lower right')

    lsm.alpha = .1
    plt.subplot(212)
    plt.xlim(0,3*pi)
    plt.ylim(-2,2)
    plt.legend()
    plt.plot(x,y, 'ro', ms=10)
    plt.plot(tics,sin(tics), linewidth=lw, label="sin")
    lsm.fit(x,y)
    plt.plot(tics,lsm.predict(tics), linewidth=lw, label="fitting");
    plt.legend(loc = 'lower right')

    plt.show()


if __name__ == "__main__":
    main()



