I could arrange each layers activations as a matrix, where each row is a layer, 
but the only issue is that the weights cant be stored as a 3d matrix (or can they?)

wait i literally can. 

so 1 layer is represented completely by:

[a1
 a2
 a3
 ... 
 an], [a1.1 a1.2 a1.3 a1.4 a1.5 ... a1.k
       a2.1 a2.2 a2.3 ... a2.k
	   ...
	   an.1 an.2 an.3 ... an.k], and
[b1
 b2
 b3
 ...
 bn]

| A  |    |Weights
|:--:|
| a1 |
| a2 |
| ...|
| an |

where **n** is the number of neurons in the layer,
and **k** is the number of neurons in the previous layer.

this can then be turned 3-d by stacking the vectors into a 2d one, 
and stacking the matrices into a 3-d one? maybe?

okay no im doing it oop






``` python
dataset = [sample]*200
batchSize = 5

batches = divide(dataset,batchSize) # len(batches) = batchSize, len(batches[0]) = 200/5
epochs = 1000
for i in range(epochs):
	for trainingBatch in batches:
		#the number of times this runs is called the number of iterations for one epoch
 		updateWeightsAndBiases(trainingBatch)
```