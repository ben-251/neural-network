import logic
import numpy as np
import bentests as bt
from layer import *
from network import Network
from DataHandler import *

class Ignore(bt.testCase):
	...

class Maths(bt.testCase):
	def testRelu(self):
		startingMatrix = np.array([[1, -2], [3, 0]])
		result = logic.relu(startingMatrix)
		bt.assertEquals(result, np.array([[1, 0], [3, 0]])) 

	def testToBitZero(self):
		n = to_bit(0.1)
		bt.assertEquals(n, 0.0)

	def testToBitHalfway(self):
		n = to_bit(0.5)
		bt.assertEquals(n, 1)

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
		bt.assertEquals(
			first_hidden_layer.weights,
			np.array(
				[[0.0, 0.0], [0.0 , 0.0], [0.0, 0.0], [0.0, 0.0]]
			)
		)

class NetworkTests(bt.testCase):
	def testInputLayerActivationTypes(self):
		network = Network((1,2))
		bt.assertEquals(network.layers[0].activations, np.array([[0.0]], dtype=float))

	def testHiddenLayerActivationTypes(self):
		network = Network((1,2))
		bt.assertEquals(network.layers[1].activations, np.array([[0.0], [0.0]], dtype=float))

	def testHiddenLayerWeights(self):
		network = Network((3,2))
		bt.assertEquals(
			network.layers[1].weights,
			np.array([
				[0.0, 0.0, 0.0],
				[0.0, 0.0, 0.0]
			])	
		)

	def testFeedForward(self):
		...
	
	def testInputLayer(self):
		network = Network((2,3,4))
		batch_size = 4
		data_handler = DataHandler()
		sample = data_handler.read_samples()[0]
		network.setInputLayer(sample.inputs)
		bt.assertEquals(network.layers[0].activations, np.array([[0.15077396917696806],[0.16554892655204556]]))


class DataHandlerTests(bt.testCase):
	'''
	DON'T RUN IF YOU KEEP THE TEST DATA UNCHANGED
	'''
	def testSampleInputType(self):
		data_handler = DataHandler(training_size=10,testing_size=10)
		data_handler.generate_data()
		data_handler.write_data()
		samples = data_handler.read_samples()
		bt.assertEquals(type(samples[0].inputs), tuple)
	
	def testSampleOutputType(self):
		data_handler = DataHandler(training_size=10,testing_size=10)
		data_handler.generate_data()
		data_handler.write_data()
		samples = data_handler.read_samples()
		bt.assertEquals(type(samples[0].result), float)		

	def testSampleOutputRange(self):
		data_handler = DataHandler(training_size=10,testing_size=10)
		data_handler.generate_data()
		data_handler.write_data()
		samples = data_handler.read_samples()
		bt.assertEquals(samples[0].result == 0.0 or samples[0].result == 1.0, True)

bt.test_all(
	Maths,
	Layers,
	NetworkTests,
)
