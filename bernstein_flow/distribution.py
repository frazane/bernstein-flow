from collections import OrderedDict

import jax
import jax.numpy as jnp
import distrax
from jaxtyping import Array, Float
import tensorflow_probability.substrates.jax.bijectors as tfb

from bernstein_flow.bijector import BernsteinBijector

class BernsteinFlow(distrax.Transformed):

    def __init__(self, params: Float[Array, "params"], constrained=False):
        
        batch_shape = params.shape[:-1]
        bijector = [
            tfb.Invert(tfb.SoftClip(low=0., hinge_softness=3.0)),
            tfb.Scale(jax.nn.softplus(params[..., 0])),
            tfb.Shift(params[...,1]),
            tfb.SoftClip(low=0 + 1e-4, high=1 - 1e-4, hinge_softness=1.5),
            BernsteinBijector(params[...,2:-2], constrained=constrained),
            tfb.Scale(jax.nn.softplus(params[...,-2])),
            tfb.Shift(params[...,-1]),
        ]

        super().__init__(
            distribution = distrax.Normal(jnp.zeros(batch_shape), jnp.ones(batch_shape)), 
            bijector = distrax.Inverse(distrax.Chain(bijector[::-1]))
            )