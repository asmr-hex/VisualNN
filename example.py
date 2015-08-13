from VisualNN import *
import numpy as np

arch = '(0,0:1)->(1,0:1)->(2,0)'
#f = lambda X: X

#arch = '(0,0:3)->(1,0:2)->(2,0:15)->(3,0:1)'
f = lambda X: 1./(1. + np.exp(-X))
df = lambda X: np.exp(-X)/(1. + np.exp(-X))

NN = NeuralNetwork(arch, f, df, verbose=True)
NN.feedForward(np.array([0.5,0.7]))
print NN.getOutput()
NN.backPropagation(np.array([3.]))
NN.updateWeights(.1)
NN.feedForward(np.array([0.5,0.7]))
print NN.getOutput()

#NN.illustrate()