import typing as tp

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .bijectors import Bernstein, Scale, Shift, Softclip, Chain, Invert
from .distributions import StandardNormal

class BernsteinFlow:
    """A normalizing flow based on Bernstein polynomials.
    
    This normalizing flow uses a chain of bijectors to transform a base, standard 
    normal distribution into a more complex distribution. 
    
    Forward transformation: Normal -> Complex
    Inverse transformation: Complex -> Normal
    """

    def __init__(self, params: Array, constrained: bool=False):
        """Initializes the flow.
        
        Args:
            params: A sequence of parameters for the bijectors. The first two parameters
                are used for the Scale and Shift bijectors, the last two parameters are
                used for the Scale and Shift bijectors at the end of the chain, and the
                remaining parameters are used for the Bernstein bijector.
            constrained: Whether or not theta coefficients of the Bernstein polynomial 
                are constrained to be monotonically increasing.
        
        """
        self.batch_shape = params.shape[:-1]
        self.bijector = Invert(
            Chain(
                [   
                    Invert(Softclip(low=0., hinge_softness=3.0)),
                    Scale(jax.nn.softplus(params[...,0])),
                    Shift(params[...,1]),
                    Softclip(low=0 + 1e-4, high=1 - 1e-4, hinge_softness=1.5),
                    Bernstein(params[...,2:-2], constrained=constrained),
                    Scale(jax.nn.softplus(params[...,-2])),
                    Shift(params[...,-1]),
                ][::-1]
            )
        )

        self.base_dist = StandardNormal()

    def sample(self, key: Array, num_samples: int):
        """Samples from the complex distribution.
        
        Args:
            key: A PRNG key.
            num_samples: The number of samples to generate.
        """
        return self.bijector.forward(self.base_dist.sample(key, (num_samples, *self.batch_shape)))
    
    def log_prob(self, value: ArrayLike):
        """Computes the log probability of the complex distribution.
        
        It does so by transforming the value to the standard normal distribution
        and computing the log probability there. The log probability is then
        adjusted by the log determinant of the Jacobian of the transformation, which accounts
        for the "deformation" of the probability density.

        Args:
            value: A value for which to compute the log probability.
        """
        x, ildj_y = self.bijector.inverse_and_log_det(value)
        lp_x = self.base_dist.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y
    
    def prob(self, value: ArrayLike):
        """Computes the probability of the complex distribution.
        
        Args:
            value: A value for which to compute the probability.
        """
        return jnp.exp(self.log_prob(value))