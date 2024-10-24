import numpy as np

def relu(numpyArray):
	return np.maximum(0.0, numpyArray)

def to_bit(n:float) -> bool:
	if n < 0 or n > 1:
		raise ValueError("n must be from 0 - 1")
	else:
		return n >= 0.5

def xor(a:float,b:float) -> float:
	a = to_bit(a) # could really just say "round(a)." I'm leaving it for now because of error handling
	b = to_bit(b)
	return float(bool(a^b))