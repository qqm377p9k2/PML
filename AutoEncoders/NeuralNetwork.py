import matplotlib.pyplot as plt
import __builtin__ as base
from numpy import *
from numpy.random import randn, rand, permutation
from numpy import linalg as LA

#Sigmoid activation function and its derivative
def sigmoid(x):
    return 1/(1+exp(-x))
def sigmoidDot(x):
    return sigmoid(x)*sigmoid(-x)

#Rectified Linerar activation function and its derivative
def ReL(x):
    return max(0, x)
def ReLDot(x):
    return x>0

class LabeledData(object):
    def __init__(training, test=None):
        assert((training['labels'].ndim==2)&(training['data'].ndim==2))
        assert(training['labels'].shape[0]==training['data'].shape[0])
        self.training['data']   = training['data']
        self.training['labels'] = training['labels']
        if test:
            assert((test['labels'].ndim==2)&(test['data'].ndim==2))
            assert(test['labels'].shape[0]==test['data'].shape[0])
            self.test['data']   = test['data']
            self.test['labels'] = test['labels']

    def generateBatches(self, learningModel):
        labels= self.training['labels']
        data  = self.training['data']
        iodim = learningModel.ioDimension()
        assert((labels.shape[1]==iodim['output'])&(data.shape[1]==iodim['input']))
        datasz = labels.shape[0]
        assert(mod(datasz, learningModel.batchsz)==0)
        noBatches = datasz/learningModel.batchsz
        order = permutation(datasz)
        labels= reshape(labels[order], [noBatches , batchsz, labels.shape[1]])
        data  = reshape(data[order],   [noBatches , batchsz, data.shape[1]])
        return zip(data, labels)
    
class Discriminants(object):
    def ioDimension(self):
        assert(False)
        return {'input':0, 'output':0}

class batchLearningAlgorithms(object):
    def __init__(self, batchsz=100):
        self.batchsz = batchsz

class NeuralNetwork(batchLearningAlgorithms, Discriminants):
    def __init__(self, noUnits, learningParams):
        super(NeuralNetwork, self).__init__()
        self.noUnits = noUnits #the Num Of Units
        self.learningParams = learningParams
        #noUnits = noUnits + [1, 0] #adding a pseudo visible unit for biasing
        self.weights = []
        self.biases = []
        self.activationFun = {'':sigmoid, 'derivative':sigmoidDot}
        for layer in self.layers():
            self.weights.append(0.01*randn(noUnits[layer], noUnits[layer-1]))
            self.biases.append(0.01*randn(noUnits[layer],1))
        
    def train(self, labeledData, noEpochs):
        assert(noEpochs>0)
        learningRate = learningParams['learningRate']
        decayingRate = decayingParams['decayingRate']
        epochs = range(noEpochs)
        derivatives = {'weights':[zeros(W.shape) for W in self.weights], 
                       'biases': [zeros(b.shape) for b in self.biases]}
        for epoch in epochs:
            batches = labeledData.generateBatches(self)
            for batch in batches:
                self.computeDerivative(*batches, derivatives)
                for layer in self.layers():
                    self.weights[layer-1]= self.weights[layer-1]+ learningRate*derivatives['weights'][layer-1]
                    self.biases[layer-1] = self.biases[layer-1] + learningRate*derivatives['biases'][layer-1]                    

    def computeDerivative(self, labels, data, derivatives):
        activations, stimuli = self.computeForwardPass(data)#a: activation, z:stimuli in the lecture note by Ng
        errors = self.computeBackwardPass(labels, activations, stimuli)
        for layer in self.layers():
            derivatives['weights'][layer-1]= dot(errors[layer], activations[layer-1].T)
            derivatives['biases'][layer-1] = mean(errors[layer], axis=1)[:,newaxis]


    def computeForwardPass(self, data):
        assert(data.ndim==2)
        assert(data.shape[0]==self.noUnits[0])
        activations = [data]
        stimuli = [array([[]])]
        for layer in self.layers():
            stimuli.append(dot(self.weights[layer-1], activations[layer-1]) + self.biases[layer-1])
            activations.append(self.activationFun[''](stimuli[layer]))
        return activations, stimuli


    def computeBackwardPass(self, labels, activations, stimuli):
        assert(activations[-1].ndim==2)
        assert(activations[-1].shape[0]==self.noUnits[-1])
        assert(all(activations[-1].shape==labels.shape))
        errors = [array([[]])]*(self.noLayers()-1)
        errors[-1] = -(activations[-1] - labels)*self.activationFun['derivative'](stimuli[-1])
        for layer in self.layers(direction=-1):
            errors[layer] = dot(self.weights[layer].T, errors[layer+1])*self.activationFun['derivative'](stimuli[layer])
        return errors

    def layers(self, direction=1):
        if(direction==1):
            return range(1,len(self.noUnits))
        elif(direction==-1):
            return range(len(self.noUnits)-2,0,-1)
        else:
            assert(False)

    def ioDimension(self):
        return {'input':noUnits[0], 'output':noUnits[-1]}

    def noLayers(self):
        return len(self.noUnits)+1
        
    def features(self, data):
        pass

def testComputeForwardPass(nn):
    a,z = nn.computeForwardPass(rand(5,1))
    print([foo.shape for foo in a])
    print([foo.shape for foo in z])
    return a,z

def testComputeBackwardPass(nn,a,z):
    d = nn.computeBackwardPass(array([[1],[0]]),a,z)
    print([foo.shape for foo in d])

def testLayers(nn):
    print(nn.layers())
    print(nn.layers(direction=-1))

def testComputeDerivative():
    nn = NeuralNetwork([5,3,5,2], {'learningRate': 1e-4})
    der = {'weights':[zeros(W.shape) for W in nn.weights], 
           'biases': [zeros(b.shape) for b in nn.biases]}
    nn.computeDerivative(array([[1],[0]]), rand(5,1), der)
    print([foo.shape for foo in der['weights']])
    print([foo.shape for foo in der['biases']])
    print(der)
    
def main():
    nn = NeuralNetwork([5,3,5,2], {'learningRate': 1e-4})
    print(nn.weights)
    print(nn.biases)
    testLayers(nn)
    a,z = testComputeForwardPass(nn)
    testComputeBackwardPass(nn,a,z)
    testComputeDerivative()

if __name__=="__main__":
    main()
