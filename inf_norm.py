"""Tests the calculation of the infinity norm.
"""
import jax.numpy as jnp

import numpy as np

Z = jnp.array([[12.0, -12.0], [15.0, -15.0], [18.0, -18.0]])
rowwise_inf_norm = jnp.linalg.norm(Z, ord="inf", axis=1)
rowwise_inf_norm = jnp.linalg.norm(Z, ord=jnp.inf, axis=1)
print(f"{rowwise_inf_norm=}")

Z_np = np.array([[12.0, -12.0], [15.0, -15.0], [18.0, -18.0]])
np_rowwise_inf_norm = np.linalg.norm(Z, ord=0.3, axis=1)
print(f"{np_rowwise_inf_norm=}")
