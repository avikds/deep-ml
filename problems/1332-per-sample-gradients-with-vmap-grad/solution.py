import jax
import jax.numpy as jnp

def per_sample_grads(w, b, X, y):
    """Per-example gradients of (w·x + b - y)^2."""

    def loss_single(w, b, x, target):
        residual = jnp.dot(w, x) + b - target
        return residual ** 2

    grad_fn = jax.grad(loss_single, argnums=(0, 1))

    dW, db = jax.vmap(
        grad_fn,
        in_axes=(None, None, 0, 0)
    )(w, b, X, y)

    return dW, db