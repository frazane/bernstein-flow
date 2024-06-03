import typing as tp

from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import ndtri


class StandardNormal:

    def sample(self, key: Array, shape: tp.Sequence[int] = ()):
        return jr.normal(key, shape=shape)
    
    def log_prob(self, value: ArrayLike) -> Array:
        return -0.5 * (value**2 + jnp.log(2 * jnp.pi))
    
    def prob(self, value: ArrayLike) -> Array:
        return jnp.exp(self.log_prob(value))
    
    def quantile(self, q: ArrayLike) -> Array:
        return ndtri(q)
