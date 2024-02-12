import jax
import jax.numpy as jnp
from jax.scipy.stats import beta
import distrax
from jaxtyping import Array, Float


class BernsteinBijector(distrax.Bijector):
    """Initializes a Bernstein bijector."""

    def __init__(self, thetas: Float[Array, "batch thetas"], constrained: bool = False):
        self.thetas = thetas if constrained else constrain_thetas(thetas)
        super().__init__(event_ndims_in=0, event_ndims_out=0)

    def forward(self, x: Float[Array, "sample"]) -> Float[Array, "batch sample"]:
        """Computes y = f(x)."""
        bernstein_poly = get_bernstein_poly(self.thetas)
        clip = 1e-7
        x = jnp.clip(x, clip, 1.0 - clip)
        return bernstein_poly(x)

    def forward_log_det_jacobian(self, x: Float[Array, "sample"]) -> Float[Array, "batch sample"]:
        """Computes log|det J(f)(x)|."""
        bernstein_poly = get_bernstein_poly_jac(self.thetas)
        clip = 1e-7
        x = jnp.clip(x, clip, 1.0 - clip)
        return jnp.log(bernstein_poly(x))

    def inverse(self, y: Float[Array, "sample"]) -> Float[Array, "batch sample"]:
        """Computes x = f^{-1}(y)."""
        batch_shape = self.thetas.shape[:-1]
        n_points = 100
        clip = 1e-7
        x_fit = jnp.linspace(clip, 1 - clip, n_points)[:,None] * jnp.ones(
            (1,) + batch_shape
        )
        y_fit = self.forward(x_fit)

        def inp(y, y_fit, x_fit):
            return jnp.interp(y, y_fit, x_fit)

        x = jax.vmap(inp, in_axes=-1, out_axes=-1)(y, y_fit, x_fit)
        return x

    def forward_and_log_det(self, x):
        """Computes y = f(x) and log|det J(f)(x)|."""
        y = self.forward(x)
        logdet = self.forward_log_det_jacobian(x)
        return y, logdet

    def inverse_and_log_det(self, y):
        """Computes y = f(x) and log|det J(f)(x)|."""
        y = self.inverse(y)
        logdet = 0
        return y, logdet
    


def get_beta_params(order):
    alpha = jnp.arange(1., order + 1.)
    beta = alpha[::-1]
    return alpha, beta


def get_bernstein_poly(theta):
    theta_shape = theta.shape
    order = theta_shape[-1]
    beta_params = get_beta_params(order)

    def bernstein_poly(x):
        bx = beta.pdf(x[...,jnp.newaxis], *beta_params)
        z = jnp.mean(bx * theta, axis=-1)
        return z

    return bernstein_poly


def get_beta_derivative_params(order):
    alpha = jnp.arange(1., order)
    beta = alpha[::-1]
    return alpha, beta


def get_bernstein_poly_jac(theta):
    theta_shape = theta.shape
    order = theta_shape[-1]

    beta_der_params = get_beta_derivative_params(order)

    def bernstein_poly_jac(y):
        by = beta.pdf(y[..., jnp.newaxis], *beta_der_params)
        dtheta = theta[..., 1:] - theta[..., 0:-1]
        dz_dy = jnp.sum(by * dtheta, axis=-1)
        return dz_dy

    return bernstein_poly_jac


def constrain_thetas(theta_unconstrained, fn=jax.nn.softplus):

    d = jnp.concatenate(
        (
            jnp.zeros_like(theta_unconstrained[..., :1]),
            theta_unconstrained[..., :1],
            fn(theta_unconstrained[..., 1:]) + 1e-4,
        ),
        axis=-1,
    )

    return jnp.cumsum(d[..., 1:], axis=-1)