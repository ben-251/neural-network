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
	2.1. Feedforward (counter-intuitively, this doesn't have to actually happen _to_ the network. It just has to compute all the activations to then use to backpropagate. It's just a function. So a `network` object doesn't need an `activations` property, for example)
	2.2. Find the average {derivative of the cost function w.r.t each of the weights and biases} across all training examples within that minibatch 
	2.3. Update each weight and bias by that derivative
