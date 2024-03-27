import jax.numpy as jnp

def softplus(x):
    return jnp.log(1 + jnp.exp(x))

def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1)

def softplus_derivative(x):
    expx = jnp.exp(x)
    return expx / (1 + expx)


def softclip(x, low=0.0, high=1.0):
    hl = high - low
    return - softplus(hl - softplus(x - low)) * hl / softplus(hl) + high

def inverse_softclip(y, low=0.0, high=1.0):
    return jnp.log(-jnp.exp(low) + jnp.exp(high) / (jnp.exp(((high - y) * jnp.log(jnp.exp(high) * jnp.exp(-low) + 1) / (high - low))) - 1))

def softclip_derivative(x, low=0.0, high=1.0):
    hl = high - low
    numerator = hl * jnp.exp(high - 2*low + x)
    denominator = (jnp.exp(-low + x) + 1) * (jnp.exp(hl) + jnp.exp(-low + x) + 1) * jnp.log(jnp.exp(hl) + 1)
    return numerator / denominator