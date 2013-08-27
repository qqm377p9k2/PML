import numpy as np
import random
import math
import operator
import matplotlib.pyplot as plt

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def sampleDirichlet(params):
    sample = [random.gammavariate(a,1) for a in params]
    return [v/sum(sample) for v in sample]

def PDFDirichlet(mu, params):
    """
    the probability density function 
    of the dirichlet distribution
    """
    assert(len(mu)==len(params))
    pstr = prod(map(math.pow, mu, 
                    map(lambda x: x-1, params)))
    Z = prod([math.gamma(a) for a in params])/math.gamma(sum(params))
    return pstr/Z


def npSampleDirichlet(params, size=None):
    params = np.array(params)
    if len(np.shape(params))==1:
        if size==None:
            sample = np.array([np.random.gamma(a,1.0) for a in params])
        else:
            if np.shape(params)[0]==size[0]:
                sample = np.array([np.random.gamma(a,1.0,size[1]) for a in params])
            elif np.shape(params)[0]<size[0]:
                sample = np.array([np.random.gamma(a,1.0,size[1]) for a in params[:-1]])
                rest = 1+size[0]-np.shape(params)[0]
                sample = np.concatenate((sample, 
                                         np.random.gamma(params[-1],1.0,
                                                         (rest,size[1]))))
            else:
                assert(False)
    elif len(np.shape(params))==0:
        assert(size!=None)
        sample = np.random.gamma(params, 1.0, size)
    return sample/np.sum(sample, axis=0)

vgamma = np.vectorize(math.gamma)

def factorial(n):
    """
    Compute the factorial of a number
    This function is used to test the accuracy of logGamma function
    """
    def fact(i, prod): 
        if i > 0:
            return fact(i-1, prod*i)
        else:
            return prod
    return fact(n,1)

def logGamma(x):
    """approximate log(gamma(x)) for large x by using Stirling's approximation """
    if x < 170:
        return math.log(math.gamma(x))
    else:
        return (x-1)*math.log(x-1) - (x-1) + 0.5*math.log(2*math.pi*(x-1))
      
def npPDFDirichlet(mu, params):
    mu = np.array(mu)
    params = np.array(params)
    assert(len(mu)==len(params))
    assert(sum(mu)==1)
    pstr = np.product(mu**(params-1))
    Z = np.product(vgamma(params))/math.gamma(sum(params))
    return pstr/Z

def npLogPDFDirichlet(mu, params):
    mu = np.array(mu)
    params = np.array(params)
    assert(len(mu)==len(params))
    assert(sum(mu)==1)
    logPstr = np.sum((params-1)*np.log(mu))
    logZ = np.sum(log(vgamma(params))) - logGamma(np.sum(params))
    return logPstr - logZ


def main():
    #
    print(logGamma(501)-math.log(factorial(500)))
    #
    noSamples = 300;
    plt.subplots_adjust(hspace=0.4)
    tran = np.array([np.array([1,-1,0])/np.sqrt(2),np.array([-1,-1,2])/np.sqrt(6)])
    #alpha = 0.1
    plt.subplot(231)
    samples = npSampleDirichlet(0.1,(3,noSamples))
    samples = np.dot(tran, samples)
    plt.scatter(samples[0,:], samples[1,:])
    #alpha = 1.0
    plt.subplot(232)
    samples = npSampleDirichlet(1,(3,noSamples))
    samples = np.dot(tran, samples)
    plt.scatter(samples[0,:], samples[1,:])
    #alpha = 10.0
    plt.subplot(233)
    samples = npSampleDirichlet(10,(3,noSamples))
    samples = np.dot(tran, samples)
    plt.scatter(samples[0,:], samples[1,:])
    #
    plt.subplot(234)
    samples = npSampleDirichlet([2,1],(3,noSamples))
    samples = np.dot(tran, samples)
    plt.scatter(samples[0,:], samples[1,:])
    #
    plt.subplot(235)
    samples = npSampleDirichlet([1,2,1],(3,noSamples))
    samples = np.dot(tran, samples)
    plt.scatter(samples[0,:], samples[1,:])
    #
    plt.subplot(236)
    samples = npSampleDirichlet([1,1,2],(3,noSamples))
    samples = np.dot(tran, samples)
    plt.scatter(samples[0,:], samples[1,:])
    #
    plt.show()



if __name__=="__main__":
    main()





