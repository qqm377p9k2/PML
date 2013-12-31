import __builtin__ as base
from numpy import *
import functools as ftk

import sys

import time
from pylab import *
from matplotlib.patches import Ellipse

from numpy.random import randn, rand, permutation, gamma, standard_cauchy
from numpy import linalg as LA

import MNIST
from Data import *
from rbm import *
from variedParam import * 

from GaussianMixtures import *
from basics import *

def generateData():
    gmm = GaussianMixture(N=1000)
    trans= array([1.5*sqrt(6),0.])
    corr = array([[6.,0.],
                  [0.,0.1]])
    delta = pi/14.0
    main = pi/4.0

    rot = rotation2D(-main -delta)
    corr1 = dot(dot(rot, corr), rot.T)
    trans1= dot(trans, rot)
    gmm.append(normalDist(trans1, corr1), 0.5)
    rot = rotation2D(-main +delta)
    corr2 = dot(dot(rot, corr), rot.T)
    trans2= dot(trans, rot)
    gmm.append(normalDist(trans2, corr2))

    (t,x) = gmm.sample().mixtures()
    data = DataSet(training = Data(data = x, labels=t))
    return data

    #plt.show()

colorTable = ['blue', 'red', 'purple', 'orange']

def drawData(generator=generateData):
    fig = figure()
    ax = fig.add_subplot(111, aspect='equal')
    data = generator()
    t = data.training.labels
    x = data.training.data
    print(x.shape)
    colors = [colorTable[int(label)] for label in t]
    #x -= mean(x, axis=0)
    #x += ones(2)/(dot(x,ones(2))[:,newaxis] + 1)
    ax.add_artist(scatter(x[:,0], x[:,1], color=colors))
    #ax.set_xlim(-15, 15)
    #ax.set_ylim(-15, 15)        
    show()
    fig = figure()
    hist(x[:,0], 50, normed=1, histtype='step')
    show()


def monitorInit():
    ion()
    plt.hold(False)    
    pass

@static_var("fig", figure())
def monitor(rbm, data):
    fig = monitor.fig
    ax = fig.add_subplot(111, aspect='equal')
    #canvas = ax.figure.canvas
    #background = canvas.copy_from_bbox(ax.bbox)
    #canvas.restore_region(background)
    #
    t = data.training.labels
    x = data.training.data
    colors = [colorTable[int(label)] for label in t]
    #
    #H = asarray([[0,0],[0,1],[1,0],[1,1]])
    H = asarray(possible_configs(rbm.shape[0]))
    centers = rbm.b + rbm.sigma * dot(H,rbm.W)
    prob = unnorm_probability_of_h(rbm, H)
    prob = prob/sum(prob)
    print(prob)
    print(rbm.activationProb(data))
    ells = [Ellipse(xy=center, width=3*rbm.sigma[0], height=3*rbm.sigma[1], angle=0) for center in centers]
    #
    #sample = rbm.sample()
    #labels = ['green']*(sample.shape[0])
    sample = rbm.particles#concatenate((sample,rbm.particles),axis=0)
    labels = ['green']*(rbm.particles.shape[0]) #+= ['purple']*(rbm.particles.shape[0])
    #
    x = concatenate((x,sample), axis=0)
    colors += labels
    ax.add_artist(scatter(x[:,0], x[:,1], color=colors))
    for vec in rbm.sigma*rbm.W:
        ax.add_artist(arrow(rbm.b[0], rbm.b[1], *vec, head_width=0.3, head_length=0.5))
    for e,p in zip(ells, prob):
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(p)
        e.set_facecolor(zeros(3))
    ax.set_xlim(-5, 10)
    ax.set_ylim(-7, 7)        
    draw()
    show()
    print(rbm.W)
    print(rbm.b)
    print(rbm.a)
    #time.sleep(0.01)
    

def unnorm_probability_of_h(rbm, h):
    centers = rbm.b + rbm.sigma * dot(h,rbm.W)
    foo = (sum((centers/rbm.sigma)**2, axis=1)/2)[:,newaxis] + dot(h,rbm.a[:,newaxis])
    #print(foo)
    return exp(foo)

def drawTest():
    rbm = GRBM(M=2, N=2)
    rbm.W = asarray([[-1,2],[3,1]])
    rbm.b = asarray([5,5])
    rbm.a = asarray([-7,-23])
    rbm.sigma = 0.7*asarray([1,1])
    #
    H = asarray([[0,0],[0,1],[1,0],[1,1]])
    centers = rbm.b + rbm.sigma * dot(H,rbm.W)
    prob = unnorm_probability_of_h(rbm, H)
    prob = prob/sum(prob)
    print(prob)
    ells = [Ellipse(xy=center, width=3*rbm.sigma[0], height=3*rbm.sigma[1], angle=0) for center in centers]
    sample = rbm.sample()
    #
    fig = figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(sample[:,0], sample[:,1], color='green')
    for e,p in zip(ells, prob):
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(p)
        e.set_facecolor(zeros(3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    show()

def generateData2(dist = 10, N=1000, delta = pi/10.0, ratio=0.4):
    '''GM with four components where two of them are at the same position'''
    gmms = [GaussianMixture(N=N), GaussianMixture(N=N)]
    for gmm in gmms:
        gmm.append(normalDist(zeros(2), asarray([[1.,0],[0,1.]])), ratio)
        gmm.append(normalDist(asarray([dist,0]), asarray([[1.,0],[0,1.]])))
    samples = [gmm.sample().mixtures()[1] for gmm in gmms]
    labels = [zeros(gmms[0].noDataPoints), ones(gmms[1].noDataPoints)]
    
    main = 0*pi/4.0

    rot = rotation2D(-main -delta)
    samples[0] = dot(samples[0], rot)
    rot = rotation2D(-main +delta)
    samples[1] = dot(samples[1], rot)

    samples = concatenate(samples, axis = 0)
    labels = concatenate(labels, axis = 1)
    data = DataSet(training = Data(data = samples, labels=labels))
    return data
    

def generateData3():
    '''distributions that GRBM can model'''
    dists = 0.2*asarray([6, 3])
    gmms = [GaussianMixture(N=1000), GaussianMixture(N=1000)]
    for gmm,dist in zip(gmms, dists):
        gmm.append(normalDist(zeros(2), asarray([[0.1,0],[0,0.1]])), 0.5)
        gmm.append(normalDist(asarray([dist,0]), asarray([[0.1,0],[0,0.1]])))
    samples = [gmm.sample().mixtures()[1] for gmm in gmms]
    labels = [zeros(gmms[0].noDataPoints), ones(gmms[1].noDataPoints)]
    
    delta = pi/2.0
    main = 0*pi/4.0

    rot = rotation2D(-main)
    samples[0] = dot(samples[0], rot)
    rot = rotation2D(-main +delta)
    samples[1] = dot(samples[1], rot) + dists/2

    samples = concatenate(samples, axis = 0)
    #samples -= asarray([6,0])
    labels = concatenate(labels, axis = 1)
    data = DataSet(training = Data(data = samples, labels=labels))
    return data

def generateData4():
    '''crude V with lines'''
    dim = asarray([3, 1.2])
    delta = 0.5
    N = 500
    mixture = [comp*dim-asarray([delta, dim[1]/2.0]) for comp in [rand(N,2), rand(N,2)]]

    samples = mixture
    labels  = [zeros(mixture[0].shape[0]), ones(mixture[1].shape[0])]
    
    delta = pi/8.0
    main = 0*pi/4.0

    rot = rotation2D(-main - delta)
    samples[0] = dot(samples[0], rot)
    rot = rotation2D(-main +delta)
    samples[1] = dot(samples[1], rot)

    samples = concatenate(samples, axis = 0)
    #samples -= asarray([6,0])
    labels = concatenate(labels, axis = 1)
    data = DataSet(training = Data(data = samples, labels=labels))
    return data


def generateData5():
    '''V with overlap on the tip'''
    dim = asarray([6, 2])
    N = 1000
    theta = pi/10.0
    main = 0*pi/4.0

    samples = [comp*dim for comp in [rand(N,2), rand(N,2)]]
    
    flags   = [dot(comp, asarray([-tan(2*theta), 1]))< 0 for comp in samples]
    samples = [comp[flag] for comp,flag in zip(samples, flags)] 
    samples = [comp[0:500] for comp in samples]
    labels  = [zeros(samples[0].shape[0]), ones(samples[1].shape[0])]
    samples = [samples[0]*asarray([1,-1]), samples[1]]

    rot = rotation2D(-main -theta)
    samples[0] = dot(samples[0], rot)
    rot = rotation2D(-main +theta)
    samples[1] = dot(samples[1], rot)

    samples = concatenate(samples, axis = 0)
    #samples -= asarray([6,0])
    labels = concatenate(labels, axis = 1)
    data = DataSet(training = Data(data = samples, labels=labels))
    return data

def generateData6():
    '''V without overlap'''
    dim = asarray([7, 2])
    N = 1000
    theta = pi/6.0
    main = 0*pi/4.0

    samples = [comp*dim for comp in [rand(N,2), rand(N,2)]]
    
    flags   = [dot(comp, asarray([-tan(theta), 1]))< 0 for comp in samples]
    samples = [comp[flag] for comp,flag in zip(samples, flags)] 
    samples = [comp[0:500] for comp in samples]
    labels  = [zeros(samples[0].shape[0]), ones(samples[1].shape[0])]
    samples = [samples[0]*asarray([1,-1]), samples[1]]

    rot = rotation2D(-main -theta)
    samples[0] = dot(samples[0], rot)
    rot = rotation2D(-main +theta)
    samples[1] = dot(samples[1], rot)

    samples = concatenate(samples, axis = 0)
    #samples -= asarray([6,0])
    labels = concatenate(labels, axis = 1)
    data = DataSet(training = Data(data = samples, labels=labels))
    return data

def f(x):
    return sigmoid(10*(x-0.5))

def generateData7():

    dim = asarray([6, 2])
    N = 1000
    theta = pi/10.0
    main = 0*pi/4.0


    samples = [comp*dim for comp in [rand(N,2), rand(N,2)]]
    for i in xrange(2):
        samples[i][:,1] = dim[1]*(samples[i][:,1]/dim[1])**(3)
        #samples[i][:,0] = dim[0]*f(samples[i][:,0]/dim[0])
    flags   = [dot(comp, asarray([-tan(2*theta), 1]))< 0 for comp in samples]
    samples = [comp[flag] for comp,flag in zip(samples, flags)] 
    samples = [comp[0:500] for comp in samples]
    labels  = [zeros(samples[0].shape[0]), ones(samples[1].shape[0])]
    samples = [samples[0]*asarray([1,-1]), samples[1]]

    rot = rotation2D(-main -theta)
    samples[0] = dot(samples[0], rot)
    rot = rotation2D(-main +theta)
    samples[1] = dot(samples[1], rot)

    samples = concatenate(samples, axis = 0)
    #samples -= asarray([6,0])
    labels = concatenate(labels, axis = 1)
    data = DataSet(training = Data(data = samples, labels=labels))
    return data

##non mixtures
def generateData10():
    '''DistortedNorm'''
    dist = 10
    gmm = GaussianMixture(N=1000)
    gmm.append(normalDist(zeros(2), asarray([[5,0],[0,0.5]])))
    samples = gmm.sample().mixtures()[1] 
    samples -= asarray([samples[:,0].min(), 0])
    samples[:,0] = samples[:,0].max()*(samples[:,0]/samples[:,0].max())**(4)
    labels = zeros(gmm.noDataPoints)
    
    delta = 0*pi/10.0
    main = 0*pi/4.0

    rot = rotation2D(-main -delta)
    samples = dot(samples, rot)

    data = DataSet(training = Data(data = samples, labels=labels))
    return data

def generateData11():
    '''uniform-Gamma dist'''
    shape = asarray([3, 1])
    dist = 10
    N = 1000
    samples = asarray([gamma(0.1, size=N), rand(N)]).T
    samples *= shape
    labels = zeros(N)
    
    delta = 0*pi/10.0
    main = 0*pi/4.0

    rot = rotation2D(-main -delta)
    samples = dot(samples, rot)

    data = DataSet(training = Data(data = samples, labels=labels))
    return data


def pointsInATriangle(size = 10000, shape = asarray([3, 5]), theta = pi/6.0, 
                      dist = ftk.partial(gamma, shape=0.1), limit=100):
    '''uniform-Gamma dist in a triangle'''
    flags = ones(size, dtype=bool)
    samples = zeros((size, 2))

    while any(flags):
        N = sum(flags)
        samples[flags] = (asarray([dist(size=N), rand(N)]).T - asarray([0,0.5]))*shape
        flags   = ~((dot(samples, asarray([-tan(theta), 1]))<0) & (dot(samples, asarray([tan(theta), 1]))>0) & (samples[:,0]<limit))

    labels = zeros(samples.shape[0])
    
    main = 0*pi/4.0

    rot = rotation2D(-main)
    samples = dot(samples, rot)

    data = DataSet(training = Data(data = samples, labels=labels))
    return data

generateData12 = pointsInATriangle
generateData13 = ftk.partial(pointsInATriangle, shape = asarray([1,5]), theta=pi/8.0, dist = ftk.partial(gamma, shape=2))
generateData14 = ftk.partial(pointsInATriangle, shape = asarray([5,5]), theta=pi/8.0, dist=lambda size: rand(size))
generateData15 = ftk.partial(pointsInATriangle, shape = asarray([1.5,5]), theta=pi/8.0, dist = standard_cauchy, limit=9, size=10000)
generateData16 = ftk.partial(pointsInATriangle, shape = asarray([4,5]), theta=pi/12.0, dist = standard_cauchy, limit=9, size=10000)
generateData17 = ftk.partial(pointsInATriangle, shape = asarray([4,10]), theta=pi/12.0, dist = lambda size: (exp(10*rand(size))-1)/10, limit=5, size=10000)
generateData18 = ftk.partial(pointsInATriangle, shape = asarray([2.5,3.5]), theta=pi/8.0, dist = ftk.partial(gamma, shape=1), limit=5)

def generateData20(shape =  asarray([3,0.7]), main = 0):
    N = 1000
    samples = randn(N, 2)*shape 
    delta = pi/14.0
    trans = asarray([5,0])

    labels = zeros(samples.shape[0])
    rot = rotation2D(main)
    trans= dot(trans, rot)
    samples = dot(samples, rot) + trans

    data = DataSet(training = Data(data = samples, labels=labels))
    return data


def generateData21(N = 5000, sigma = 1., shape =  asarray([10,5]), main = 0, ratios=[8,2,2,1]):
    ratios = asarray(ratios, dtype = float)
    ratios /= sum(ratios)

    gmm = GaussianMixture(N=N)

    cov = asarray([[sigma,0],[0,sigma]])
    means = asarray([[0,0], 
                     [shape[0]/2.0, shape[1]/2.0],
                     [shape[0]/2.0, -shape[1]/2.0],
                     [shape[0], 0]])
    for mean,ratio in zip(means[:-1], ratios[:-1]):
        gmm.append(normalDist(mean, cov), ratio)

    gmm.append(normalDist(means[-1], cov))

    #print(gmm.noDataPoints)
    #print(gmm._GaussianMixture__noDataPoints)
    labels, samples = gmm.sample().mixtures()
    #labels = zeros(samples.shape[0])
    rot = rotation2D(main)
    samples = dot(samples, rot)

    data = DataSet(training = Data(data = samples, labels=labels))
    return data

generateData22 = ftk.partial(generateData21, shape=asarray([20,6]), ratios=[1,1,1,1])
generateData23 = ftk.partial(generateData21, shape=asarray([15,6]), ratios=[90,5,5,1])
generateData24 = ftk.partial(generateData21, N = 10000, shape=asarray([7,3]), ratios=[80,5,5,1])
generateData25 = ftk.partial(generateData2,  dist = 4, N=5000, delta = pi/10.0, ratio=0.85) #not good
generateData26 = ftk.partial(generateData2,  dist = 4, N=5000, delta = pi/15.0, ratio=0.85) #neither
#The data dist needs to be close to the dist class that GRBM can efficiently model
generateData27 = ftk.partial(generateData21, N = 10000, shape=asarray([8,2.5]), ratios=[80,5,5,1]) 
generateData28 = ftk.partial(generateData21, N = 10000, shape=asarray([10,2.5]), ratios=[80,5,5,1]) 
generateData29 = ftk.partial(generateData21, N = 10000, shape=asarray([8,2.2]), ratios=[80,5,5,2]) 
generateData30 = ftk.partial(generateData21, N = 10000, shape=asarray([8,2.2]), ratios=[75,5,5,5]) #bad, too densy on the end
generateData31 = ftk.partial(generateData21, N = 10000, shape=asarray([7,2.7]), ratios=[80,5,5,1])
generateData32 = ftk.partial(generateData21, N = 10000, shape=asarray([7,2.8]), ratios=[80,5,5,1])
generateData33 = ftk.partial(generateData21, N = 10000, shape=asarray([7,2.75]), ratios=[80,5,5,1])
generateData34 = ftk.partial(generateData21, N = 10000, shape=asarray([7,2.6]), ratios=[80,5,5,1])
generateData35 = ftk.partial(generateData21, N = 10000, shape=asarray([7,2.6]), ratios=[80,5,5,0.01])
generateData36 = ftk.partial(generateData21, N = 10000, shape=asarray([7,2.6]), ratios=[80,5,5,0.1])


def parse_command_line_args(argv, default_values):
    assert(len(argv)<=len(default_values))
    argv += ['']*(len(default_values) - len(argv))
    return [arg if arg else val for  arg, val in zip(argv, default_values)]


def main(generator = generateData, save={'filename':False}):
    epochs = 5000
    #epochs = 30 
    monitorInit()
    data = generator()
    rbm = GRBM(M=4, N=2)
    #rbm.lrate = variedParam(0.02)
    rbm.sparsity = {'strength': .5, 'target': 0.05}
    rbm.lrate = variedParam(0.02, schedule=[['linearlyDecayFor', epochs]])
    rbm.mom   = variedParam(0.0)

    rbm.initWithData(data)
    #rbm.sigma = 0.4*rbm.sigma
    rbm.sigma = sqrt(1.0)*asarray([1.0,1.0])
    rbm.CDN = 1
    rbm.setAlgorithm('PCD')
    monitor(rbm, data)
    if save['filename']:
        with gzip.open(save['filename'], 'wb') as output:
            logger=genLogger(output, save['interval'])
            rbm.train(data, epochs, monitor=monitor, logger=logger)
    else:
        logger=emptyLogger
        rbm.train(data, epochs, monitor=monitor, logger=logger)

if __name__ == "__main__":
    generator, test, save = parse_command_line_args(sys.argv[1:], ['generateData12', 'False', ''])
    save = {'filename':save, 'interval':10}
    if eval(test): 
        drawData(eval(generator))
    else:
        main(eval(generator), save=save)
