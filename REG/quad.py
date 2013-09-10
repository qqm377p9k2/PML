import __builtin__ as base
from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

import LSM

def main():
    (x,y) = LSM.data()
    (xwn, ywn) = LSM.data(Noise=True)
    lsm = LSM.LSM_L2(6)
    lsm.fit(x,y)
    lsm2 = LSM.LSM_L2(6)
    lsm2.fit(xwn,ywn)
    tics = 2*pi*arange(0,1,0.02);
    lw = 5 #line width
    plt.subplot(211)
    plt.legend()
    plt.plot(x,y, 'ro', ms=10)
    plt.plot(tics,sin(tics), linewidth=lw, label="sin")
    plt.plot(tics,lsm.predict(tics), linewidth=lw, label="fitting");
    plt.legend(loc = 'lower right')

    plt.subplot(212)
    plt.xlim(0,2*pi)
    plt.ylim(-2,2)
    plt.legend()
    plt.plot(xwn,ywn, 'ro', ms=10)
    plt.plot(tics,sin(tics), linewidth=lw, label="sin")
    plt.plot(tics,lsm2.predict(tics), linewidth=lw, label="fitting");
    plt.legend(loc = 'lower right')
    plt.show()


if __name__ == "__main__":
    main()
