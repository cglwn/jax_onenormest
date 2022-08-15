from functools import partial

from jax import jit
import jax.numpy as jnp

def are_parallel(u, v):
    n = u.shape[0]
    dot_prod = u.dot(v)
    return dot_prod in [n, -n]

if __name__ == "__main__":
    u = jnp.array([1, 1])
    v = jnp.array([1, -1])
    jit(are_parallel)(u, v)
