import numpy as np
from numpy import arange, transpose, ix_, r_, c_, linalg, random

import matplotlib.pyplot as plt


def sample(n):
    return linalg.qr(random.randn(n,n))[0]

def test():
    fig = plt.figure()
    cmap = plt.get_cmap('rainbow')
    ax = fig.gca()
    N = 20
    bases = [sample(2) for i in  xrange(N)]
    print bases
    for i,base in enumerate(bases):
        color = cmap(i/float(N))
        print color
        print [plt.plot(*transpose([[0,0], p]), color=color) for p in base]
    ax.set_xlim(-1.01, 1.01)
    ax.set_ylim(-1.01, 1.01)
    plt.show()
