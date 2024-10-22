I don't intend on freestyling this, giving up halfway, then leaving crumbs for my future self to pick up. Not this time, at least. so this is a step-by-step plan, which I will of course expand a bunch of times. I'm doing this primarily because the "index-chasing" nature of this project will likely obscure the real complexity, so I want as much control over my inevitable confusion as possible. 

# Network structure
let `n` be the length of layer_a, and `k` be the length of layer_a+1:
- Layers: never written explicitly as a property.
	- previous activations: nx1 (column vec)
	- next activations: kx1 (col vec)
- weights: kxn matrix, where n
- biases: nx1 (column vector)
(quick check: kxn x nx1 + kx1 -> kx1)
- nonlin function (and it's derivative): just a property abstracted (with default of Relu)

- training data: pair of "question" and "expected answer"
(that said, I think since the backprop process does look back at previous layers 
as part of the chain rule calculation, it might be necessary to store all the activations after all)

# Levels of training process
## Level 0:
1. Split training data into minibatches to stochastically decrease the gradient
2. For each mini batch, run the backpropagation routine and update every weight and bias once.

## Level 1:
(not nesting numbering because that would get lengthy)
1. Split training data into minibatches to stochastically decrease the gradient
2. For each mini batch:
	2.1. Feedforward (counter-intuitively, this doesn't have to actually happen _to_ the network. It just has to compute all the activations to then use to backpropagate. It's just a function. So a `network` object doesn't need an `activations` property, for example. // perhaps, I'll have to expand the calculation to see that this idea works)
	2.2. Find the average {derivative of the cost function w.r.t each of the weights and biases} across all training examples within that minibatch 
	2.3. Update each weight and bias by that derivative

## Level 2—Expanding the Calculations
#### part 1: data prep 
1.  Generate training data as pairs of inputs (the two neurons) and expected outputs (representing 1 and 0)
2. Split training data into smaller arrays of equal size (the last one can be one or two less than the rest, to prevent issues with primes)

#### part 2: feed-forward
1. to generate the nth vector of activations, simply compute `non_lin(w*a_{n-1} + b)`, where `non-lin(v)` applies the non-linearity to each entry, and `n >= 1` (the layers are zero indexed). repeat until you've reached the output neurons

#### part 3: the derivative with respect to each weight in layer L (to j from k):
Cost for a single training example, $C_0$ will be the sum of the MSE costs for each j 
since for one neuron layers, $\partial C / \partial a^{(L-1)} = w^{(L)} \times$ `deriv_non_lin`$(Z^{(L)}) \times 2(a^{(L)}-y)$, then for multiple neurons, I can just either use $2(a^{(L)}-y)$ or the sum calculation for $\partial C / \partial a^{(l)}$, depending on whether there are multiple activations feeding into $a^{(L)}$. Putting that all together, the derivative works out to be the product of the following:
1. $\partial z_j / \partial w_{jk}$ — activation k on layer L-1.
2. $\partial a_j /\partial z_j$ — derivative of non_lin of weighted sum, z_j for layer L
3. $\partial C_0 / \partial a_j$ — One of:

	a. the sum of all the derivative computations (the cost function w.r.t each weight to j from k in the next layer (L+1))
	b.  $2(a-y)$

the question remains, though, of how to get these values from vectors. For example, when finding deriv_non_lin(z_j), we'd have to recompute wa+b for j, but without the non_lin_function. and functions like relu wouldn't have an inverse, so i can't just invert the non_linearity on the activations. I'll come back to this



