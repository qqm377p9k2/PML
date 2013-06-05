import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from numpy import sin, cos
from numpy.random import rand, randn
from numpy import linalg as LA

def mkBasis(x):
    return lambda m: np.power(x,m);

N = 50;
x = 2*np.pi*rand(N);
y = np.sin(x) + 0.3*randn(N);
y[0] = y[0]+3;
y[5] = y[5]-10;

tics = 2*np.pi*np.arange(0,1,0.02);
phi = np.mat(map(mkBasis(x), np.arange(1,6)));

L1=0
I =L1*np.identity(5) ;
wml0 = (LA.inv(I+(phi*np.transpose(phi)))*phi) *np.transpose(np.mat(y));
xx0 = np.array(np.transpose(wml0)*np.mat(map(mkBasis(tics), np.arange(1,6))));

L1=2
I =L1*np.identity(5) ;
wml = (LA.inv(I+(phi*np.transpose(phi)))*phi) *np.transpose(np.mat(y));
xx = np.array(np.transpose(wml)*np.mat(map(mkBasis(tics), np.arange(1,6))));

L1=4
I =L1*np.identity(5) ;
wml2 = (LA.inv(I+(phi*np.transpose(phi)))*phi) *np.transpose(np.mat(y));
xx2 = np.array(np.transpose(wml2)*np.mat(map(mkBasis(tics), np.arange(1,6))))

L1=6
I =L1*np.identity(5) ;
wml3 = (LA.inv(I+(phi*np.transpose(phi)))*phi) *np.transpose(np.mat(y));
xx3 = np.array(np.transpose(wml3)*np.mat(map(mkBasis(tics), np.arange(1,6))));

L1=8
I =L1*np.identity(5) ;
wml4 = (LA.inv(I+(phi*np.transpose(phi)))*phi) *np.transpose(np.mat(y));
xx4 = np.array(np.transpose(wml4)*np.mat(map(mkBasis(tics), np.arange(1,6))));

L1=10
I =L1*np.identity(5) ;
wml5 = (LA.inv(I+(phi*np.transpose(phi)))*phi) *np.transpose(np.mat(y));
xx5 = np.array(np.transpose(wml5)*np.mat(map(mkBasis(tics), np.arange(1,6))));

#draw results
fig1 = plt.figure()
lw = 2 #line width
plt.xlim(0,2*np.pi)
plt.ylim(-5,3)
plt.legend()
plt.plot(x,y, 'ro', ms=10)
plt.plot(tics,np.sin(tics), linewidth=lw, label="sin")
plt.plot(tics,xx0[0], linewidth=lw, label="lambda=0");
plt.plot(tics,xx[0], linewidth=lw, label="lambda=2");
plt.plot(tics,xx2[0], linewidth=lw, label="lambda=4");
plt.plot(tics,xx3[0], linewidth=lw, label="lambda=6");
plt.plot(tics,xx4[0], linewidth=lw, label="lambda=8");
plt.plot(tics,xx5[0], linewidth=lw, label="lambda=10");
plt.legend(loc = 'lower right')
plt.show()
#print(y)
#print(xx[0])

quit()
