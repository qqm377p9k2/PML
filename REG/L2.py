import __builtin__ as base
from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

import LSM

def main():
    (x,y) = LSM.data()
    (xwn, ywn) = LSM.data(Noise=True)
    lsm = LSM.LSM_L2(M=7, l=2.)
    tics = 2*pi*arange(0,1,0.02);
    lw = 5 #line width

    ls = power(7., range(-2,3))
    print(ls)
    for idcs in range(len(ls)):
        plt.subplot(len(ls),1,idcs+1)
        plt.xlim(0,2*pi+2)
        plt.ylim(-2,2)
        plt.legend()
        plt.plot(xwn,ywn, 'ro', ms=10)
        plt.plot(tics,sin(tics), linewidth=lw, label="sin")
        lsm.l = ls[idcs]
        lsm.fit(xwn,ywn)
        plt.plot(tics,lsm.predict(tics), linewidth=lw, label='l={:.1}'.format(ls[idcs]));
        plt.legend(loc = 'lower right')
    plt.show()


if __name__ == "__main__":
    main()
