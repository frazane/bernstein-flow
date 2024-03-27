import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import ndtri


class StandardNormal:

    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype

    def sample(self, key, shape: tuple = ()):
        return jr.normal(key, shape=shape, dtype=self.dtype)
    
    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * (value**2 + jnp.log(2 * jnp.pi))
    
    def prob(self, value: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(self.log_prob(value))
    
    def quantile(self, q: jnp.ndarray | float) -> jnp.ndarray:
        return ndtri(q)
