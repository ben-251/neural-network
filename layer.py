import numpy as np
from maths import *

class Layer:
	def __init__(self, layer_size: int | None = None):
		if layer_size is None:
			layer_size = 10
		self.size: int = layer_size
		self.activations: np.ndarray = np.zeros((layer_size,1))

class WeightedLayer(Layer):	
	def __init__(self,prevLayer: Layer, layer_size: int | None = None):
		# rememer its not x,y but "number of rows x number of columns"
		super().__init__(layer_size=layer_size)
		self.weights: np.ndarray = np.zeros((self.size, prevLayer.size), dtype=float) #TODO: make Weights and biases randomised
		self.biases: np.ndarray = np.zeros((self.size), dtype=float)
	
	def updateActivations(self, previous_layer: Layer):
		self.activations = relu(np.dot(self.weights, previous_layer.activations) + self.biases)
	
	def storeWeightsAndBiases(self):
		...

class InputLayer(Layer):
	def __init__(self, layer_size: int | None = None):
		super().__init__(layer_size = layer_size)
	
	def getInputActivations(self):
		...