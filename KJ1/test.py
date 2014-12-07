from Basics.utils import pickle, unpickle, measuring_speed
from Basics.numpy_ import *
import matplotlib.pyplot as plt

tint = arange(0, 50, 0.1)



def first(tint, target = 2.):
    t = 0.
    x = zeros(len(tint))
    for i, dt in enumerate(np.diff(tint)):
        t += dt
        x[i+1] = x[i] + dt*(1-x[i]/target)
    return x

def riccati0(tint, params=[1., 1.,1]):
    alpha, beta, gamma= params
    t = 0.
    x = zeros(len(tint))
    for i, dt in enumerate(np.diff(tint)):
        t += dt
        x[i+1] = x[i] + dt*(gamma + alpha*x[i] - beta*x[i]**2)
    return x
    

#x = first(tint)
x = riccati0(tint, [1, 1, 1e-5])
plt.plot(tint, x)
plt.show()
