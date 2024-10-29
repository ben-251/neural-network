I don't intend on freestyling this, giving up halfway, then leaving crumbs for my future self to pick up. Not this time, at least. so this is a step-by-step plan, which I will of course expand a bunch of times. I'm doing this primarily because the "index-chasing" nature of this project will likely obscure the real complexity, so I want as much control over my inevitable confusion as possible. 

# Network structure
## Ideas
let `n` be the length of layer_a, and `k` be the length of layer_a+1:
- Layers: never written explicitly as a property.
	- previous activations: nx1 (column vec)
	- next activations: kx1 (col vec)
- weights: kxn matrix, where n
- biases: kx1 (column vector)
(quick check: kxn x nx1 + kx1 -> kx1)
- nonlin function (and it's derivative): just a property abstracted (with default of Relu)

- training data: pair of "question" and "expected answer"
(that said, I think since the backprop process does look back at previous layers 
as part of the chain rule calculation, it might be necessary to store all the activations after all)

## Structure
class Network:
	weights, list of kxn matrices, where k is the length of layer L, and n is the length of layer L-1
	biases, list of np nx1 vectors
	activations: list of np nx1 vectors, computed as non_lin(w*a_prev + b)



# Levels of training process
## Level 0:
1. Split training data into minibatches to stochastically decrease the gradient
2. For each mini batch, run the backpropagation routine and update every weight and bias once.

## Level 1:
(not nesting numbering because that would get lengthy)
1. Split training data into minibatches to stochastically decrease the gradient
2. For each mini batch:
	2.1. Feedforward (counter-intuitively, this doesn't have to actually happen _to_ the network. It just has to compute all the activations to then use to backpropagate. It's just a function. So a `network` object doesn't need an `activations` property, for example. // perhaps, I'll have to expand the calculation to see that this idea works) // what I should have said is that I don't need `layer` objects/properties/classes. not that I don't need the activations. Not quite sure what happened with my logic there.
	2.2. Find the average {derivative of the cost function w.r.t each of the weights and biases} across all training examples within that minibatch 
	2.3. Update each weight and bias by that derivative

## Level 2—Expanding the Calculations
#### part 1: data prep 
1.  Generate training data as pairs of inputs (the two neurons) and expected outputs (representing 1 and 0)
2. Split training data into smaller arrays of equal size (the last one can be one or two less than the rest, to prevent issues with primes)

#### part 2: feed-forward
1. to generate the nth vector of activations, simply compute `non_lin(w*a_{n-1} + b)`, where `non-lin(v)` applies the non-linearity to each entry, and `n >= 1` (the layers are zero indexed). repeat until you've reached the output neurons

1. start by taking in the inputs as a column vector, this will be the first layer in our activations list
2. to compute the next one, N(w * a + b), where a is the previous vector (initially input)

#### part 3: the derivative with respect to each weight in layer L (to j from k):
Cost for a single training example, $C_0$ will be the sum of the MSE costs for each j 
since for one neuron layers, $\partial C / \partial a^{(L-1)} = w^{(L)} \times$ `deriv_non_lin`$(Z^{(L)}) \times 2(a^{(L)}-y)$, then for multiple neurons, I can just either use $2(a^{(L)}-y)$ or the sum calculation for $\partial C / \partial a^{(l)}$, depending on whether there are multiple activations feeding into $a^{(L)}$. Putting that all together, the derivative works out to be the product of the following:
1. $\partial z_j / \partial w_{jk}$ — activation k on layer L-1.
2. $\partial a_j /\partial z_j$ — derivative of non_lin of weighted sum, z_j for layer L
3. $\partial C_0 / \partial a_j$ — One of:   
	a. the sum of all the derivative computations (the cost function w.r.t each weight to j from k in the next layer (L+1))   
	b.  $2(a-y)$

the question remains, though, of how to get these values from vectors. For example, when finding deriv_non_lin(z_j), we'd have to recompute wa+b for j, but without the non_lin_function. and functions like relu wouldn't have an inverse, so i can't just invert the non_linearity on the activations. I'll come back to this. For now I need to make sure that the calculations actually account for every weight. I also need to compute the gradient for the biases

#### Part 4: Derivative w.r.t each bias in layer L (j):
$C = (a^{(L)}-y)^2;\ a^{(L)} = N(z);\ z = wa^{(L-1)}+b$, where N is the nonlinearity function, giving:
$$\begin{equation}
\begin{align*}
	&\frac{\partial C}{\partial b_j} = \frac{\partial C}{\partial a_j} \frac{\partial a_j}{\partial z_j} \frac{\partial z_j}{\partial b_j} \\[1.5em]
	&\frac{\partial C}{\partial a_j} = 2(a-y) \text{(or the sum above)} \\[1em]
	&\frac{\partial a_j}{\partial z_j} = N'(z_j) \\[1em]
	&\frac{\partial z_j}{\partial b_j} = 1 \\[1em]
	&\frac{\partial C}{\partial b_j} =  N'(z_j) \times 2(a-y) \text{(or the obvious)}
\end{align*}
\end{equation}$$

In python that might look something like:
```python
cost_wrt_bias(j) = N_prime(z[j]) * cost_wrt_activation(j) # or something
```



