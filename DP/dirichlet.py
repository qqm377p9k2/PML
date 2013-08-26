import numpy as np
import random
import matplotlib.pyplot as plt

def sampleDirichlet(params):
    sample = [random.gammavariate(a,1) for a in params]
    return [v/sum(sample) for v in sample]

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
      
def main():
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
    plt.show()


if __name__=="__main__":
    main()





