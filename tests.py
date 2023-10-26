import maths
import numpy as np
import bentests as bt
from layer import *

class Ignore(bt.testCase):
	def testRelu(self):
		startingMatrix = np.array([[1, -2], [3, 0]])
		result = maths.relu(startingMatrix)
		bt.assertEquals(result, np.array([[1, 0], [3, 0]])) # OOH not so simple, numpy needs special treatment. 

class Maths(bt.testCase): ...

class Layers(bt.testCase):
	def testLayerSizeGiven(self):
		new_layer = Layer(layer_size=3)
		bt.assertEquals(new_layer.size,3)
	
	def testDefaultLayerSize(self):
		new_layer = Layer()
		bt.assertEquals(new_layer.size,10)

	def testDefaultWeightedLayerSize(self):
		new_layer = WeightedLayer(prevLayer=Layer())
		bt.assertEquals(new_layer.size,10)

	def testWeightedLayerShape(self):
		previous_layer = Layer(layer_size=15)
		main_layer = WeightedLayer(previous_layer,layer_size=10)
		bt.assertEquals(main_layer.weights.shape, (10,15))

	def testWeightedLayerBiasShape(self):
		previous_layer = Layer(layer_size=4)
		main_layer = WeightedLayer(previous_layer, layer_size=10)
		bt.assertEquals(main_layer.biases.shape, (10,))

	def testLayerActivations(self):
		layer = Layer(layer_size=2)
		bt.assertEquals(layer.activations, np.array([[0.0],[0.0]]))

	def testWeightedLayerActivations(self):
		input_layer = Layer(layer_size=3)
		first_hidden_layer = WeightedLayer(prevLayer=input_layer, layer_size=3)
		bt.assertEquals(first_hidden_layer.activations, np.array([[0.0],[0.0],[0.0]]))
	
	def testWeightedLayerWeights(self):
		input_layer = Layer(layer_size=2)
		first_hidden_layer = WeightedLayer(prevLayer=input_layer, layer_size=4)
		bt.assertEquals(first_hidden_layer.weights,
		np.array(
			[[0.0, 0.0], [0.0 , 0.0], [0.0, 0.0], [0.0, 0.0]]
		)
		)

bt.test_all(
	Maths,
	Layers
)