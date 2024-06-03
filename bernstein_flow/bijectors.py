import abc
import typing as tp

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax.scipy.special import gammaln

from tensorflow_probability.substrates.jax.bijectors import Softplus as SoftplusTFP

class Bijector:
    """The base bijector class.
    
    A bijector is a transformation that can be applied to a random variable.
    All bijectors must implement the following methods:
    
    - _forward: computes the forward transformation f(x).
    - _inverse: computes the inverse transformation f^{-1}(y).
    - _forward_log_det_jacobian: computes the log|det J(f)(x)|.

    """

    @abc.abstractmethod 
    def _forward(self, x):
        ...

    def forward(self, x: ArrayLike) -> Array:
        """Computes y = f(x)."""
        return self._forward(x)
    
    @abc.abstractmethod
    def _inverse(self, y):
        ...

    def inverse(self, y: ArrayLike) -> Array:
        """Computes x = f^{-1}(y)."""
        return self._inverse(y)
    
    @abc.abstractmethod
    def _forward_log_det_jacobian(self, x):
        ...


    def forward_log_det_jacobian(self, x: ArrayLike) -> Array:
        """Computes log|det J(f)(x)|."""
        return self._forward_log_det_jacobian(x)


    def inverse_log_det_jacobian(self, y: ArrayLike) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        return -self.forward_log_det_jacobian(self.inverse(y))


    def forward_and_log_det(self, x: ArrayLike) -> tuple[Array, Array]:
        """Computes f(x) and log|det J(f)(x)|."""
        return self.forward(x), self.forward_log_det_jacobian(x)


    def inverse_and_log_det(self, y: ArrayLike) -> tuple[Array, Array]:
        """Computes f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)




class Bernstein(Bijector):
    """A Bernstein polynomial bijector.
    
    The Bernstein polynomial is defined as:
    
    B(x) = sum_{i=0}^{n} theta_i * B_{i,n}(x)
    
    where B_{i,n}(x) is the i-th Bernstein basis polynomial of degree n.
    """

    def __init__(self, thetas: Array | tp.Sequence[float], constrained: bool = False):
        """Initializes the bijector.

        Args:
            thetas: A list of coefficients for the Bernstein polynomial.
        """
        
        self.thetas = jnp.asarray(thetas) if constrained else self.constrain_thetas(jnp.asarray(thetas))
        self.order = self.thetas.shape[-1]  
        self.alpha = jnp.arange(1, self.order + 1)
        self.beta = self.alpha[::-1]

    def _forward(self, x: ArrayLike) -> Array:
        x = jnp.clip(x, 1e-6, 1 - 1e-6)
        bx = beta_pdf(x[...,None], self.alpha, self.beta)
        return jnp.mean(self.thetas * bx, axis=-1)
    
    def _inverse(self, y):
        n_points = 100
        clip = 1e-6
        x_fit = jnp.linspace(clip, 1 - clip, n_points)
        # x_fit *= jnp.ones((1, *y.shape))
        y_fit = self.forward(x_fit)
        
        def interp_fn(y, y_fit, x_fit):
            return jnp.interp(y, y_fit, x_fit)
        
        # return jax.vmap(interp_fn, in_axes=-1, out_axes=-1)(y, y_fit, x_fit)
        return interp_fn(y, y_fit, x_fit)
    
    def _forward_log_det_jacobian(self, x):
        """Computes log|det J(f)(x)|."""
        x = jnp.clip(x, 1e-6, 1 - 1e-6)
        by = beta_pdf(x[...,None], self.alpha[:-1], self.beta[1:])
        dtheta = self.thetas[...,1:] - self.thetas[...,:-1]
        return jnp.log(jnp.sum(by * dtheta, axis=-1))
    
    def forward_and_log_det(self, x: ArrayLike) -> tuple[Array, Array]:
        y = self.forward(x)
        logdet = self.forward_log_det_jacobian(x)
        return y, logdet
         
    def inverse_and_log_det(self, y):
        """Computes y = f(x) and log|det J(f)(x)|."""
        y = self.inverse(y)
        logdet = 0
        return y, logdet
    
    def inverse_log_det_jacobian(self, y: ArrayLike) -> Array:
        return 0
    
    @staticmethod
    def constrain_thetas(theta):
        d = jnp.concatenate(
            (
                jnp.zeros_like(theta[...,:1]),
                theta[...,:1],
                jax.nn.softplus(theta[...,1:]) + 1e-4,
            ),
            axis=-1,
        )
        return jnp.cumsum(d[...,1:], axis=-1)

    def __repr__(self) -> str:
        return f"Bernstein(thetas={self.thetas})"
    
    
class Softclip(Bijector):
    """A softclip bijector.
    
    The softclip function is defined as:
    
    f(x) = - softplus(high - softplus(x - low)) * (high - low) / softplus(high) + high
    
    where softplus(x) = log(1 + exp(x)).
    """

    def __init__(self, low: float|None=None, high:float|None=None, hinge_softness: float | None = None):
        """Initializes the bijector.

        Args:
            low: The lower bound of the softclip function.
            high: The upper bound of the softclip function.
        """
        softplus_bj = Softplus(hinge_softness=hinge_softness)

        components = []

        if low is not None and high is not None:

            width = high - low
            components = [
                Shift(high),
                Scale(-width / (softplus_bj.forward(width))),
                softplus_bj,
                Shift(width),
                Scale(-1.0),
                softplus_bj,
                Shift(-low),
            ]
        
        elif low is not None:

            components = [Shift(low), softplus_bj, Shift(-low)]

        elif high is not None:
                
            components = [Shift(high), Scale(-1.0), softplus_bj, Scale(-1.0), Shift(-high)]

        self._low = low
        self._high = high
        self._hinge_softness = hinge_softness
        self._chain = Chain(components)
    
    # def _forward(self, x):
    #     hl = self.high - self.low
    #     _softplus = self.softplus_bj.forward 
    #     return - _softplus(hl - _softplus(x - self.low)) * hl / _softplus(hl) + self.high

    
    # def _inverse(self, y):
    #     l, h = self.low, self.high
    #     _softplus = self.softplus_bj.forward
    #     _softplus_inverse = self.softplus_bj.inverse
    #     inner_term = (h - y) * _softplus(h - l) / (h - l)
    #     return _softplus_inverse(h - l - _softplus_inverse(inner_term + l)) + l
    #     # return _softplus_inverse(h - l - _softplus_inverse(inner_term)) + l
    
    
    # def _forward_log_det_jacobian(self, x):
    #     return jnp.log(jnp.abs(softclip_derivative(x, self.low, self.high, self.hinge_softness)))
    
    def _forward(self, x):
        return self._chain.forward(x)
    
    def _inverse(self, y):
        return self._chain.inverse(y)
    
    def _forward_log_det_jacobian(self, x):
        return self._chain.forward_log_det_jacobian(x)
    
    def __repr__(self) -> str:
        return f"Softclip(low={self._low}, high={self._high})"
    
    
def softclip_derivative(x, low, high, c):
    c = c or 1.0
    exp_part_x = jnp.exp((x - low) / c)
    log_part = jnp.log(1 + exp_part_x)
    exp_part_hl = jnp.exp((high - low) / c)
    numerator_exp = jnp.exp(((-low + x) / c) + ((high - low - c * log_part) / c))
    denominator = c * (1 + exp_part_x) * (1 + jnp.exp((high - low - c * log_part) / c)) * jnp.log(1 + exp_part_hl)
    return (numerator_exp * (high - low)) / denominator


class Softplus(Bijector):
    """A softplus bijector.

    The softplus function is defined as:

    f(x) = log(c + exp(x/c)).

    where c is the hinge softness parameter.
    """

    def __init__(self, hinge_softness: float | None = None, low: float | None = None):
        """Initializes the bijector.

        Args:
            hinge_softness: The softness of the hinge.
        """
        self.hinge_softness = hinge_softness
        self.low = low

    def _forward(self, x):
        if self.hinge_softness is None:
            y = jax.nn.softplus(x)
        else:
            y = self.hinge_softness * jax.nn.softplus(x / self.hinge_softness)
        return y + self.low if self.low is not None else y
    
    def _inverse(self, y):
        y = y - self.low if self.low is not None else y
        if self.hinge_softness is None:
            return inverse_softplus(y)
        return self.hinge_softness * inverse_softplus(y / self.hinge_softness)
    
    def _forward_log_det_jacobian(self, x):
        if self.hinge_softness is not None:
            x /= self.hinge_softness
        return - jax.nn.softplus(-x)
    
    def inverse_log_det_jacobian(self, y) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        y = y - self.low if self.low is not None else y
        if self.hinge_softness is not None:
            y /= self.hinge_softness
        return -jnp.log(-jnp.expm1(-y))
    
    def __repr__(self) -> str:
        return f"Softplus(hinge_softness={self.hinge_softness}, low={self.low})"
    
    

class Scale(Bijector):
    """A scaling bijector.
    
    This bijector scales the input by a constant factor.
    
    Example:
    >>> x = jnp.array([0.1, 1.0, 2.3])
    >>> f = Scale(2.0)
    >>> f.forward(x)  # y = 2 * x
    Array([0.2, 2.0, 4.6], dtype=float32)
    """

    def __init__(self, scale):
        self.scale = scale

    def _forward(self, x):
        return x * self.scale
    
    def _inverse(self, y):
        return y / self.scale
    
    def _forward_log_det_jacobian(self, x):
        return jnp.log(jnp.abs(self.scale))
    
    def __repr__(self):
        return f"Scale({self.scale})"
    

class Shift(Bijector):
    """A shifting bijector.
    
    This bijector shifts the input by a constant factor.
    
    Example:
    >>> x = jnp.array([0.1, 1.0, 2.3])
    >>> f = Shift(1.0)
    >>> f.forward(x)  # y = x + 1
    Array([1.1, 2.0, 3.3], dtype=float32)"""

    def __init__(self, shift):
        self.shift = shift

    def _forward(self, x):
        return x + self.shift
    
    def _inverse(self, y):
        return y - self.shift
    
    def _forward_log_det_jacobian(self, x):
        return jnp.zeros_like(x)
    
    def __repr__(self):
        return f"Shift({self.shift})"
    

class Chain(Bijector):
    """A chain of bijectors.
    
    This class represents a chain of bijectors, which is itself a bijector.
    
    Bijectors are applied in reverse order: for two bijectors f and g, if the chain
    is [f, g], then the forward transformation is f(g(x)), not 'f and then g', therefore
    g is applied first and f is applied second.

    Example:
    >>> x = jnp.linspace(-1, 1, 1000)
    >>> f = Scale(2.0)
    >>> g = Shift(1.0)
    >>> chain = Chain([f, g])
    >>> y = chain.forward(x)  # equivalent to f(g(x))

    """

    def __init__(self, bijectors):
        self.bijectors = bijectors

    def _forward(self, x):
        y = x
        for bijector in self.bijectors[::-1]:
            y = bijector.forward(y)
        return y
    
    def _inverse(self, y):
        x = y
        for bijector in self.bijectors:
            x = bijector.inverse(x)
        return x
    
    def _forward_log_det_jacobian(self, x):
        fldj = jnp.zeros_like(x)
        for bijector in self.bijectors[::-1]:
            fldj += bijector.forward_log_det_jacobian(x)
            x = bijector.forward(x)
        return fldj
    
    def inverse_log_det_jacobian(self, y):
        """Computes log|det J(f^{-1})(y)|."""
        ildj = jnp.zeros_like(y)
        for bijector in self.bijectors:
            ildj += bijector.inverse_log_det_jacobian(y)
            y = bijector.inverse(y)
        return ildj
    
# Invert(Chain([Softclip(0.3, 1.8, hinge_softness=1.5), Shift(0.3)])).inverse(0.5)
# tfb.Invert(tfb.Chain([tfb.Softclip(0.3, 1.8, hinge_softness=1.5), tfb.Shift(0.3)])).inverse(0.5)

class Invert(Bijector):
    """An inverted bijector.
    
    This class represents the inverse of a bijector.
    
    Example:
    >>> x = jnp.linspace(-1, 1, 1000)
    >>> f = Scale(2.0)
    >>> inv = Invert(f)
    >>> y = inv.forward(x)  # equivalent to f.inverse(x)
    """

    def __init__(self, bijector):
        self.bijector = bijector

    def _forward(self, x):
        return self.bijector.inverse(x)
    
    def _inverse(self, y):
        return self.bijector.forward(y)
    
    def _forward_log_det_jacobian(self, x):
        return self.bijector.inverse_log_det_jacobian(x)
    
    def inverse_log_det_jacobian(self, y):
        """Computes log|det J(f^{-1})(y)|."""
        return self.bijector.forward_log_det_jacobian(y)
    
    

# ----------------------------------------
# Softplus
# ----------------------------------------
def softplus(x):
    return jnp.log(1 + jnp.exp(x))

# def inverse_softplus(x):
#     return jnp.log(-jnp.expm1(-x)) + x

def inverse_softplus(x):
    return jnp.log(jnp.expm1(x))

def softplus_derivative(x):
    expx = jnp.exp(x)
    return expx / (1 + expx)

# ----------------------------------------
# Softclip
# ----------------------------------------
def softclip(x, low=0.0, high=1.0):
    hl = high - low
    return - softplus(hl - softplus(x - low)) * hl / softplus(hl) + high

def inverse_softclip(z, low=0.0, high=1.0):
    hl = high - low
    inner_term = (high - z) * softplus(hl) / hl
    return inverse_softplus(high - low - inverse_softplus(inner_term)) + low

# def softclip_derivative(x, low=0.0, high=1.0):
#     hl = high - low
#     numerator = hl * jnp.exp(high - 2*low + x)
#     denominator = (jnp.exp(-low + x) + 1) * (jnp.exp(hl) + jnp.exp(-low + x) + 1) * jnp.log(jnp.exp(hl) + 1)
#     return numerator / denominator

# ----------------------------------------
# Beta distribution (needed for Bernstein bijector)
# ----------------------------------------
def beta_function(alpha, beta):
    return jnp.exp(gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))

def beta_pdf(x, alpha, beta):
    alpha, beta = map(jnp.asarray, (alpha, beta))
    B = beta_function(alpha, beta)
    return x**(alpha - 1) * (1 - x)**(beta - 1) / B

def beta_pdf_derivative(x, alpha, beta):
    alpha, beta = map(jnp.asarray, (alpha, beta))
    B = beta_function(alpha, beta)
    return ((alpha - 1) * x**(alpha - 2) * (1 - x)**(beta - 1) - (beta - 1) * x**(alpha - 1) * (1 - x)**(beta - 2)) / B
