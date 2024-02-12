# JAX implementation of Bernstein Flows

This repository contains a small JAX version of Bernstein polynomials as normalizing flows, see original publication [here](https://arxiv.org/abs/2004.00464). They are implemented as a combination of distrax and tensorflow probability (jax substrate) objects. Flax is then used to implement a simple probabilistic regression model that fits complex distributions.

## References:
- [1] https://github.com/kaijennissen/Normalizing_Flows
- [2] https://github.com/MArpogaus/TensorFlow-Probability-Bernstein-Polynomial-Bijector