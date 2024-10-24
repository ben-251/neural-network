from itertools import pairwise
from typing import Iterable, List, Optional
from DataHandler import *
from layer import *
import numpy as np

class Network:
	def __init__(self,layer_sizes: List[int], inputNeurons:Optional[List[float]]=None, learning_rate:float|None=None, isTesting:bool|None=None, isZeroed: bool|None = None):
		'''
		Creates a Simple Network with each layer size determining the number of total layers,
		where the first is an input layer and the last is an output one.

		The learning rate determines the size of each step of the gradient descent during backpropagation
		`isTesting` sets up the network to train or test the model. default behaviour is training
		'''
		self.initialise_layers(layer_sizes,inputNeurons,isZeroed=isZeroed)
		self.learning_rate = learning_rate
		self.isTesting = True if isTesting else False # Kept this way to handle none automatically
	
	def initialise_layers(self, layer_sizes: List[int], inputNeurons:Optional[List[float]]=None, isZeroed=None):
		'''
		Creates all layers in the network based on the sizes given.
		'''
		if isZeroed is None:
			isZeroed = False
		if inputNeurons is None:
			inputNeurons = [0.0]*layer_sizes[0]

		self.weights = self.generate_weights(layer_sizes)
		self.biases = self.generate_biases(layer_sizes)
		self.activations = self.initialise_activations(layer_sizes, inputNeurons)

	def generate_weights(self, layer_sizes: List[int]):
		weights = []
		for previous_layer_size, next_layer_size in pairwise(layer_sizes):
			# to next (k) from prev (n), so kxn (by "structured analysis.md") 
			random_weights = np.random.randint(0, 10, size=(
				next_layer_size, previous_layer_size
			))
			weights.append(random_weights)
		return weights

	def initialise_activations(self, layer_sizes, input_neurons):
		activations = []
		first_activations = np.array(input_neurons).reshape(-1,1) # make a column vector out of the activations
		activations.append(first_activations)
		for layer_size in layer_sizes[1:]:
			# Trying to decide whether this should feedforward initially.
			# I feel like yes, but then in test data when does it start training officially. 
			# I'll need to look at the rest of the code. For now i'll assume no, since that's less work.
			new_activations = np.zeros(
				layer_size
			)
			activations.append(new_activations)
		return activations

	def generate_biases(self, layer_sizes: List[int]) -> List[np.ndarray]:
		biases = []
		for layer_size in layer_sizes[1:]: # the first layer shouldn't have biases
			random_biases = np.random.randint(0,10, size = (layer_size, 1))
			biases.append(random_biases)
		return biases

	def set_input_layer(self, inputs:Iterable[float]):
		self.activations[0] = np.array(inputs).reshape(-1,1)

	def storeWeightsAndBiases(self):
		FILE_PATH = "data/weight and bias data.json"
		data_handler = DataHandler()
		with open(FILE_PATH, "w") as f:
			entries = []
			for i, (weights, biases) in enumerate(zip(self.weights, self.biases)):
				new_entry = {}
				new_entry["layer_number"] = i
				new_entry["weights"] = data_handler.encode_matrix(weights)
				new_entry["biases"] = data_handler.encode_matrix(biases)
				entries.append(new_entry)
			f.write(json.dumps(entries,indent=4))

	def loadWeightsAndBiases(self):
		FILE_PATH = "data/weight and bias data.json"
		data_handler = DataHandler()
		with open(FILE_PATH, "r") as f:
			layer_data = json.load(f)

			self.weights = [
				data_handler.decode_matrix(layer_datum["weights"])
				for layer_datum in layer_data
			]
			self.biases = [
				data_handler.decode_matrix(layer_datum["biases"])
				for layer_datum in layer_data
			]

	def feedforward(self):
		# set each layer of activations to non_lin(wa+b)
		for i in range(1, len(self.activations)):
			self.activations[i] = np.dot(self.weights[i], self.activations[i-1]) + self.biases[i] # Definitely test this, I haven't even looked through to see that the indices make sense
	
	def backpropagate(self):
		'''
		Good luck future self...
		'''
		raise NotImplementedError