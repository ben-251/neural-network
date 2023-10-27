import numpy as np
from itertools import islice

def relu(numpyArray):
	return np.maximum(0.0, numpyArray)

def to_bit(n:float) -> bool:
	if n < 0 or n > 1:
		raise ValueError("n must be from 0 - 1")
	elif n >= 0.5:
		return True
	else: 
		return False

def xor(a:float,b:float) -> float:
	a = to_bit(a)
	b = to_bit(b)
	if (a and not b) or (b and not a):
		return 1.0
	else:
		return 0.0

def batched(iterable, n):
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch