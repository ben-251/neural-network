import math
import random
from typing import Tuple, List
from maths import *

class Sample:
	def __init__(self, inputs: Tuple[float, ...], result: float):
		self.inputs = inputs
		self.result = result

class DataHandler:
	def __init__(self,training_size:int|None=None, testing_size: int|None = None) -> None:
		self.getData(training_size=training_size, testing_size=testing_size)
		# self.write_data()

	def getData(self,training_size:int|None=None, testing_size: int|None = None) -> None:
		if training_size is None:
			training_size = 60_000
		if testing_size is None:
			testing_size = 10_000
		self.training_data = self.getSamples(training_size)
		self.testing_data = self.getSamples(testing_size)	
	
	def getSamples(self, n) -> List[Sample]:
		samples: List[Sample] = []
		for _ in range(n):
			inputs = random.random(), random.random()
			answer = xor(inputs[0], inputs[1])
			samples.append(Sample(inputs,answer))
		return samples
	
	def write_data(self) -> None:
		self.write_samples(self.training_data, "data/training_data.txt")
		self.write_samples(self.testing_data, "data/test_data.txt")
	
	def write_samples(self, data: List[Sample], file_path: str) -> None:
		with open(file_path, "w") as f:
			for sample in data:
				new_entry = f"{[input_ for input_ in sample.inputs]}"[1:-1] + f"\n{sample.result}\n"
				f.write(new_entry)
	
	def read_samples(self, data_form) -> List[Sample]:
		if not data_form in ["testing", "training"]: #TODO: convert to enum
			raise ValueError("invalid data form")
		elif data_form == "testing":
			file_path = "data/test_data.txt"
		else:
			file_path = "data/training_data.txt"
		samples = []

		with open(file_path, "r") as f:
			for input_line, output_line in batched(f,2):
				inputs = tuple([float(input_number) for input_number in input_line.strip().split(", ")])
				output_line = float(output_line.strip())
				samples.append(Sample(inputs, output_line))
		return samples

