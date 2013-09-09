import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.random import rand, randn

N = 150;
training_steps = 30000
x = 2*np.pi*rand(N)
y = np.sin(x) + 0.1*randn(N)

###artifact
y[0] += 5
x[0] = 5
###

M = 5;
wml = 0.01*randn(M)
powers = np.arange(M,dtype=float)
initLrate = 1e-6
lrate = initLrate

phi = np.array([np.power(x, power) for power in powers])
print(np.shape(phi))

batchsz = 10
alpha = 0.1

for i in range(training_steps):
    #lrate = initLrate * (1000./(i+1000.))
    for datIdx in np.random.permutation(N):
        coef = y[datIdx] - np.dot(wml,phi[:, datIdx])
        #print(coef)
        #print(lrate*coef*phi[:, datIdx])
        #print(wml)
        #print(wml + lrate*coef*phi[:, datIdx])
        wml = wml + lrate*(coef*phi[:, datIdx] - alpha*np.sign(wml))
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
plt.ylim(-3,8)
plt.plot(x,y, 'ro', ms=10)
plt.plot(tics,np.sin(tics), linewidth=lw)
plt.plot(tics,guess, linewidth=lw);
plt.show()




