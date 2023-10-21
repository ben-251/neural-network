import numpy as np

class Layer:
	activations: np.ndarray

	def __init__(self, layer_size=None):
		self.size = layer_size
		self.activations = np.zeros((1,layer_size)) # not sure whats happening here

class WeightedLayer(Layer):
	weights: np.ndarray
	biases: np.ndarray
	
	def __init__(self,prevLayer, layer_size=None):
		super().__init__(self,layer_size)
		self.weights = np.zeros((prevLayer.layerSize, layer_size))
		self.biases = np.zeros((layer_size))
		