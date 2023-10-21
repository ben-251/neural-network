import numpy as np


# sandboxing
def relu(numpyArray):
    return np.maximum(0.0, numpyArray)


weights = np.array([[1, 2], [3, 4]])
prev_activations = np.array([[0.1], [0.2]])
biases = np.array([[-1], [2]])
newActivations = relu(np.dot(weights, prev_activations) + biases)
print(newActivations)


threeDmatrix = np.zeros((1, 1, 1))