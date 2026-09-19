import jax
import jax.numpy as jnp

def df_dx(x):
    """Derivative of f(x) = x**3 + 2x at float x, via jax.grad. Returns float."""
    f = lambda t: t**3 + 2 * t
    return float(jax.grad(f)(jnp.asarray(x, dtype=float)))