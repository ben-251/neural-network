from itertools import pairwise
from typing import Iterable
from DataHandler import *
from layer import *
import numpy as np

class Network:
	def __init__(self,layer_sizes: Iterable[int], learning_rate:float|None=None, isTesting:bool|None=None):
		'''
		Creates a Simple Network with each layer size determining the number of total layers,
		where the first is an input layer and the last is an output one.

		The learning rate determines the size of each step of the gradient descent during backpropagation
		`isTraining` sets up the network to train or test the model. default behaviour is training
		'''
		self.initialise_layers(layer_sizes)
		self.learning_rate = learning_rate
		self.isTesting = True if isTesting else False # Kept this way to handle none automatically
	
	def initialise_layers(self, layer_sizes: Iterable[int]):
		'''
		Creates all layers in the network based on the sizes given.
		'''
		self.layers = []
		for i, size in enumerate(layer_sizes):
			if i == 0:
				self.layers.append(Layer(layer_size=size))
			else:
				self.layers.append(WeightedLayer(self.layers[-1], layer_size=size)) # watch out for index problems

	def setInputLayer(self, inputs):
		activation_list = []
		for input_ in inputs:
			activation_list.append([input_])
		self.layers[0].activations = np.array(activation_list)

	def feedforward(self):
		for prevLayer, currentLayer in pairwise(self.layers):
			currentLayer.updateActivations(prevLayer)
	
	def backpropagate(self):
		'''
		Good luck future self...
		'''
		raise NotImplementedError