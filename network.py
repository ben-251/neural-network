from itertools import pairwise
from typing import Iterable
from layer import *

class Network:
	def __init__(self,layer_sizes: Iterable[int],learning_rate:float|None=None):
		'''
		Creates a Simple Network with each layer size determining the number of total layers,
		where the first is an input layer and the last is an output one.

		The learning rate determines the size of each step of the gradient descent during backpropagation
		'''
		self.initialise_layers(layer_sizes)
		self.learning_rate = learning_rate
	
	def initialise_layers(self, layer_sizes: Iterable[int]):
		self.layers = []
		for i, size in enumerate(layer_sizes):
			if i == 0:
				self.layers.append(Layer(layer_size=size))
			else:
				self.layers.append(WeightedLayer(self.layers[-1], layer_size=size)) # i dont think this will cause any indexing issues but thats what tests are for

	def feedforward(self):
		for prevLayer, currentLayer in pairwise(self.layers):
			currentLayer.updateActivations(prevLayer)