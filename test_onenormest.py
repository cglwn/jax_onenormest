from jax_onenormest import onenormest

import jax
import jax.numpy as jnp

def test_exit_cond_1():
  A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
  assert onenormest(A, 2, 5) == 18.0

def test_ind_hist():
  # Create a test matrix that exercises the ind_hist logic.
  
  ...

def test_can_jit():
  A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
  assert jax.jit(onenormest)(A, 2, 5) == 18.0
