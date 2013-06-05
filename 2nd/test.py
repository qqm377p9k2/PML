import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from numpy import sin, cos
from numpy.random import rand, randn
from numpy import linalg as LA

N = 50;
x = 2*np.pi*rand(N);
y = np.sin(x) + 0.3*randn(N);
#print(np.sin(np.arange(1,11)))


tics = 2*np.pi*np.arange(0,1,0.02);

phi = np.transpose(np.mat([x, np.power(x, 2), np.power(x, 3), np.power(x, 4), np.power(x, 5)]));
phid = LA.inv(np.transpose(phi)*phi)*np.transpose(phi);
wml = phid*np.transpose(np.mat(y));

xx = np.array(np.transpose(wml)*np.mat([tics, np.power(tics, 2), np.power(tics, 3), np.power(tics, 4), np.power(tics, 5)]))

#draw results
fig1 = plt.figure()
lw = 5 #line width
plt.xlim(0,2*np.pi)
plt.ylim(-1.5,1.5)
plt.plot(x,y, 'ro', ms=10)
plt.plot(tics,np.sin(tics), linewidth=lw)
plt.plot(tics,xx[0], linewidth=lw);
plt.show()
#print(y)
#print(xx[0])



