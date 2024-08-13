# ToDo:

NODE
---------
- Try with different history sizes - input sizes - frequency of training. Might be nice to train on very small time step once we have a good integrator / other input / interpolator.
- Address the problem of the high frequencies
- try new interpolator+integrator combo

- New implementation on jax



- > continuous time!!!
- > precision and computation depends on integrator !!
- > 

To solve the problem we implemented two strategies:
(1) Straight NN architecture to approximate u - > relationship (GRU, cKAN, MLP)
(2) Model problem as an ODE and learn the implicit derivative dQ'/dt . Here we first learn using a very small time step and an integrator. Then we could take learnt model for derivative, switch integrator add an interpolator for u and see if it still performs well. In this way you can trade-off computation expense and accuracy + it's a continuum model i.e. at each point t (integrator) can output a value Q given the previous points of u (here we need interpolator because our model learns relationship)

Issues:
(1) Higher frequencies are not well captured (New data? Augmentation using data from frequency domain in order to make them more explicit? For example use frequency data to adjust error of high frequencies. Possibly enforce more this time dependency in the input data using different architectures or attention?) Myabe take difference ground truth - result and do a regression w.r.t. frequencies that we are missing. 


What to do:
-- > Hamiltonian
--> attention time mechanism adjust it
--> peaks adjustment branch



TO SOLVE:
(!) backward pass of neural ODE on jax (put interpolator inside the model! No inputs! Redesign)
(!) discrepancy of the solutions of torch / jax and mse during training and at validation


THEN:
(1) Attention mechanism trick
(2) Non-uniform sampling
(3) Adaptive integration

MAYBE:
(*) Error branch with fft