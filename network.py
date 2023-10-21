from itertools import pairwise

class Network:
	def __init__(self,LayerCount):
		self.layers = [Layer()
	
	def feedforward(self):
		for prevLayer, currentLayer in pairwise(layers):
			currentLayer.updateActivations(prevLayer)