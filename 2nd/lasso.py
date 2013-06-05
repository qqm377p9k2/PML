import theano
import numpy as np
import theano.tensor as T
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.random import rand, randn
#from numpy import linalg as LA

def mkBasis(x):
    return lambda m: np.power(x,m);

M = 6;
N = 500;
x = 2*np.pi*rand(N);
y = np.mat(np.sin(x) + 0.3*randn(N));
#y[0] = y[0]+3;
#y[5] = y[5]-10;

tics = 2*np.pi*np.arange(0,1,0.02);

training_steps = 10000;
wml = 0.0001*np.mat(randn(M,1));

phi = np.mat(map(mkBasis(x), np.arange(M)+1));

for i in range(training_steps):
    imod = np.mod(i,N);
    wml = wml + 0.00000000001*np.sum(np.mat(np.array(y - np.transpose(wml)*phi)*np.array(phi)),1);
    #wml = wml + 0.01*(y[imod] - np.transpose(wml)*phi)*phi;
    #wml = wml + 0.0000001*twphi[0,0]*phi;
    #print(wml)

xx = np.array(np.transpose(wml)*np.mat(map(mkBasis(tics), np.arange(M)+1)));
y = np.array(y);

#draw results
fig1 = plt.figure()
lw = 5 #line width
plt.xlim(0,2*np.pi)
plt.ylim(-12,3)
plt.plot(x,y[0], 'ro', ms=10)
plt.plot(tics,np.sin(tics), linewidth=lw)
plt.plot(tics,xx[0], linewidth=lw);
plt.show()




