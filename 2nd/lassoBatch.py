import theano
import numpy as np
import theano.tensor as T
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.random import rand, randn
#from numpy import linalg as LA

M = 5;
N = 500;
training_steps = 30000
x = 2*np.pi*rand(N)
y = np.sin(x) + 0.1*randn(N)

wml = 0.01*randn(M)
powers = np.arange(M,dtype=float)
initLrate = 5.0e-6
lrate = initLrate

phi = np.array([np.power(x, power) for power in powers])
print(np.shape(phi))

batchsz = 10
noBatch = N/batchsz
alpha = 0.1

for i in range(training_steps):
    lrate = initLrate * (10000./(i+10000.))
    idcs = np.reshape(np.random.permutation(N), (noBatch,batchsz))
    for bIdx in range(noBatch):
        coef = y[idcs[bIdx,:]] - np.dot(wml,phi[:, idcs[bIdx,:]])
        wml = wml + lrate*(np.mean(coef*phi[:, idcs[bIdx,:]],axis=1) - alpha*np.sign(wml))
    err = np.mean((y - np.dot(wml,phi))**2) + alpha*np.sum(wml)
    if np.mod(i,100)==0:
        print(wml)
        print('err: ' + repr(err))
        if err < 0.08:
            break


#draw results
tics = 2*np.pi*np.arange(0,1,0.02);
guess = np.dot(wml, np.array([np.power(tics, power) for power in powers]))

fig1 = plt.figure()
lw = 5 #line width
plt.xlim(0,2*np.pi)
plt.ylim(-3,3)
plt.plot(x,y, 'ro', ms=10)
plt.plot(tics,np.sin(tics), linewidth=lw)
plt.plot(tics,guess, linewidth=lw);
plt.show()




