from functools import partial

from jax import jit
import jax.numpy as jnp
import jax.random as random

def are_parallel(u, v):
    n = u.shape[0]
    dot_prod = jnp.dot(u, v)
    return jnp.where(dot_prod == n, True, jnp.where(dot_prod == -n, True, False))

def has_parallel(A, v, n):
    for i in jnp.arange(A.shape[1]):
        if are_parallel(A[:, i], v, n):
            return True
    return False

def sample_pm_one(n: int):
    key = random.PRNGKey(0)
    return random.choice(key, jnp.array([-1.0, 1.0]), (n, 1))

# @partial(jit, static_argnames=("n", "t"))
def make_pm1_matrix(n, t):
    """Constructs a linearly independent matrix consisting of the first column all
    +1 and the remaining t-1 columns uniformly randomly sampled as ±1.
    """
    X = jnp.ones((n, 1))

    key = random.PRNGKey(0)
    for _ in jnp.arange(1, t):
        X_i = random.choice(key, jnp.array([-1, 1]), (n, 1))
        while has_parallel(X, X_i, n):
            X_i = sample_pm_one(n)
        X = jnp.concatenate((X, X_i), 1)
        key, subkey = random.split(key)
    return X / n

if __name__ == "__main__":
    make_pm1_matrix(3, 2)
