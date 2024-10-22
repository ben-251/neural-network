# Neural Network from Scratch (NNFS)

## Goal
To train a simple artificial neural network to `xor` two floats.

for example:
0.54, 0.2 -> 1, since `xor(1, 0) = 1`

0.02, 0,02 -> 0, since `xor(0, 0) = 0` 

# "Rules"
I think success should be defined somewhat concretely:
- I _am_ allowed to consult other projects, e.g. 3b1b's implementation of Nielsen's, as long as I am not copying code and am just using it to better understand the process. More precisely, I mean that referencing his code is only allowed to influence my `.md` analyses, not my direct code. That is of course tricky to moderate.
- I am _not_ allowed, at any point, to load, consider, or walk anywhere near a library like pytorch.
- I _am_ allowed to use numpy (and it's sub-module `linalg`) for clean linear algebra 
- And of course, I can use as muany maths resources (wolframalpha, calculators, desmos(???)) as I wish to.
This problem is declared solved if I get a success rate greater than 90% 
(If I stray from any of these, or change these, then git will of course reflect those changes, hence automatic accountability)

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