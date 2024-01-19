"""
Main.py

The main script

In this file, we define a simple feedforward neural network and demonstrates its usage


"""

from random import random
import math



#Create a neural network with a hidden layer 

def initializeNetwork(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs)], 'bias':random()} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden)], 'bias':random()} for i in range(n_outputs)]
	network.append(output_layer)
	return network

#Show the network architecture 
    
def showNetwork(nn):
    hidden_layer = nn[0]
    n_hidden = len(hidden_layer)
    first_neuron_weights = hidden_layer[0]['weights']
    n_inputs = len(first_neuron_weights)
    output_layer = nn[1]
    n_outputs = len(output_layer)
    print("Number of neurons in the hidden layer:", n_hidden)
    print("Number of neurons in the input layer:", n_inputs)
    print("Number of neurons in the output layer:", n_outputs)

	
#Calculate the network output
def neuronOutput(neuron, inputs):
    weights = neuron['weights']
    bias = neuron['bias']

    combination = bias
    for i in range(len(weights)):
      combination += weights[i] * inputs[i]
    output = 1.0/(1.0+math.exp(-combination))
    return output

#Calculations performed by the neural network from input to output
def forward_propagate(network, inputs):
    new_inputs = inputs
    nbLayers = len(network)
    for numl in range(nbLayers):
        layer = network[numl]
        nbNeurons_Layer = len(layer)
        outputs = []
        for numn in range(nbNeurons_Layer):
          neuron = layer[numn]
          neuron['output'] = neuronOutput(neuron, new_inputs)
          outputs.append(neuron['output'])
        if (numl<nbLayers-1):
          new_inputs = outputs
    return outputs
	
	
#Main script


nn = initializeNetwork(4, 5, 3)
showNetwork(nn)

inputs = [1.0, 1.0, 1.0, 1.0]

outputs = forward_propagate(nn, inputs)
print("inputs :",inputs)
print("outputs :",outputs)