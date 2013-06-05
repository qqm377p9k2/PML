import numpy as np
from numpy.random import rand

N = 100;
M = 5;
x = 2*np.pi*rand(N);
phi = np.zeros([M, N]);
print(phi)

for i in range(M):
    for j in range(N):
        phi[i,j] = pow(x[j], i+1)

print(phi)
