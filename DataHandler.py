import random
from typing import Tuple, List
from logic import *
from enum import Enum

class Sample:
	'''
	Individual test samples, for example: 
	inputs = (0.234, 0.892)
	result = 1.0
	'''
	def __init__(self, inputs: Tuple[float, ...], result: float):
		self.inputs = inputs
		self.result = result

class DataType(Enum):
	TESTING = "data/test_data.txt"
	TRAINING = "data/training_data.txt"

class DataHandler:
	'''
	Primary tool for creating, writing, and reading data for training and testing the model.
	'''
	def __init__(self,training_size:int|None=None, testing_size: int|None = None) -> None:
		if training_size is None:
			training_size = 60_000
		if testing_size is None:
			testing_size = 10_000
		self.training_size = training_size
		self.testing_size = testing_size

	def generate_data(self) -> None:
		self.training_data = self.generate_samples(self.training_size)
		self.testing_data = self.generate_samples(self.testing_size)	
	
	def generate_samples(self, n) -> List[Sample]:
		samples: List[Sample] = []
		for _ in range(n):
			inputs = random.random(), random.random()
			answer = xor(inputs[0], inputs[1])
			samples.append(Sample(inputs,answer))
		return samples
	
	def write_data(self) -> None:
		self.write_samples(self.training_data, isTraining=True)
		self.write_samples(self.testing_data, isTraining=False)
	
	def write_samples(self, data: List[Sample], isTraining:bool|None=None) -> None:
		data_type = DataType.TRAINING if isTraining else DataType.TESTING
		with open(data_type.value, "w") as f:
			for sample in data:
				new_entry = f"{[input_ for input_ in sample.inputs]}"[1:-1] + f"\n{sample.result}\n"
				f.write(new_entry)
	
	def read_samples(self, isTesting: bool|None=None) -> List[Sample]:
		'''
		Default value of isTesting is effectively false
		'''
		samples = []
		data_type = DataType.TESTING if isTesting else DataType.TRAINING
		with open(data_type.value, "r") as f:
			for input_line, output_line in batched(f,2):
				inputs = tuple([float(input_number) for input_number in input_line.strip().split(", ")])
				output_line = float(output_line.strip())
				samples.append(Sample(inputs, output_line))
		return samples

