from jax_onenormest import onenormest

import jax
import jax.numpy as jnp

A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
jax.jit(onenormest)(3, 2)
