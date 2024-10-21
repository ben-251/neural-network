# Neural Network from Scratch (NNFS)

## Goal
To train a simple artificial neural network to `xor` two floats.

for example:
0.54, 0.2 -> 1, since `xor(1, 0) = 1`

0.02, 0,02 -> 0, since `xor(0, 0) = 0` 

# Plan Updates
(21 Oct, 2024)
Going to go through my year-old ugly code and try to make sense of it, with my much more solid grasp 
now, on the relevant calculus and linear algebra. I will also consider using sympy for ~generalisation~ abstractions. 

Example:
```python
	x = symbols('x')
	f = Function('f')(x)  # f(x)

	f = sin(x)
	g = diff(f, x) 

	g.subs(x, 3) # g(3)
```

But of course with very different functions