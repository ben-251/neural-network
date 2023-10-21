import maths
import numpy as np
import bentests as bt

class Ignore(bt.testCase):
	def testRelu(self):
		startingMatrix = np.array([[1, -2], [3, 0]])
		result = maths.relu(startingMatrix)
		bt.assertEquals(result, np.array([[1, 0], [3, 0]])) # OOH not so simple, numpy needs special treatment. 

class Maths: ...

bt.test_all(Maths)