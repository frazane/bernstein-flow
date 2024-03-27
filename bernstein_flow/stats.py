import jax.numpy as jnp
from jax.scipy.special import gammaln


def beta_function(alpha, beta):
    """Compute the beta function."""
    return jnp.exp(gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))

def beta_pdf(x, alpha, beta):
    """Compute the PDF of the beta distribution."""
    alpha, beta = map(jnp.asarray, (alpha, beta))
    B = beta_function(alpha, beta)
    return x**(alpha - 1) * (1 - x)**(beta - 1) / B

def beta_pdf_derivative(x, alpha, beta):
    """Compute the derivative of the PDF of the beta distribution."""
    alpha, beta = map(jnp.asarray, (alpha, beta))
    B = beta_function(alpha, beta)
    return ((alpha - 1) * x**(alpha - 2) * (1 - x)**(beta - 1) - (beta - 1) * x**(alpha - 1) * (1 - x)**(beta - 2)) / B
