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

class NeuralNetwork(object):
    def __init__(self, noUnits, learningParams):
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
        learningRate = learningParams['learningRate']
        decayingRate = decayingParams['decayingRate']
        labels = labeledData['training']['labels']
        data = labeledData['training']['data']
        epochs = range(noEpochs)
        dWeights= [zeros(W.shape) for W in self.weights]
        dBiases = [zeros(b.shape) for b in self.biases]
        for epoch in epochs:
            #batches = 
            for batch in batches:
                activations, stimuli = self.ComputeForwardPass(batch)#a: activation, z:stimuli in the lecture note by Ng
                errors = self.ComputeBackwardPass(labels, activations, stimuli)
                for layer in self.layers():
                    dWeights[layer-1]= dot(errors[layer], activation[layer-1].T)
                    dBiases[layer-1] = mean(errors[layer], axis=1)

    def ComputeForwardPass(self, data):
        assert(data.shape[1]==self.noUnits[0])
        activation = [data]
        stimuli = [[]]
        for layer in self.layers():
            stimuli[layer].append(dot(self.weights[layer-1], activations[layer-1]) + self.biases[layer-1])
            activations[layer].append(activationFun[''](stimuli[layer]))
        return activations, stimuli


    def ComputeBackwardPass(self, labels, activation, stimuli):
        assert(activations[-1].shape[1]==self.noUnits[-1])
        errors = [[]]*self.noLayers()
        errors[-1] = -(activations[-1] - labels)*activationFun['derivative'](stimuli[-1])
        for layer in self.layers(direction=-1):
            errors[layer] = dot(self.weights[layer].T, errors[layer+1])*activationFun['derivative'](stimuli[layer])
        return errors

    def layers(self, direction=1):
        if(direction==1):
            return range(1,len(self.noUnits))
        elif(direction==-1):
            return range(len(self.noUnits)-2,0,-1)
        else:
            assert(False)

    def noLayers(self):
        return len(self.noUnits)+1
        
    def features(self, data):
        pass

def testComputeForwardPass(nn):
    a,z = ComputeForwardPass(rand(5,2))
    print(a)
    print(z)
    return a,z

def testComputeBackwardPass(nn,a,z):
    d = ComputeBackwardPass(rand(2,2))
    print(d)

def testLayers(nn):
    print(nn.layers())
    print(nn.layers(direction=-1))

def main():
    nn = NeuralNetwork([5,3,5,2], {'learningRate': 1e-4})
    print(nn.weights)
    print(nn.biases)
    testLayers(nn)
    a,z = testComputeForwardPass(nn)
    testComputeForwardPass(nn,a,z):

if __name__=="__main__":
    main()
