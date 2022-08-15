import jax
import jax.random as random
import jax.numpy as jnp

def _sample_pm_one_linearly_independent(A, n):
  """Samples a vector with elements drawn from {1, -1} that is not parallel with columns of A.
  """
  vec = _sample_pm_one(n)
  for i in jnp.arange(A.shape[1]):
    col = A[:, i]
    if _parallel(col, vec):
      return _sample_pm_one_linearly_independent(A, n)
  return vec

def pprint(s):
  ...
def _sample_pm_one(n: int):
  key = random.PRNGKey(0)
  return random.choice(key, jnp.array([-1.0, 1.0]), (n, 1))

def _construct_X(n, t):
  """Constructs the X matrix.
  """
  X = jnp.ones((n, 1))

  key = random.PRNGKey(0)
  for i in jnp.arange(1, t):
    X_i = random.choice(key, jnp.array([-1, 1]), (n, 1))
    while _has_parallel(X, X_i):
      X_i = _sample_pm_one(n)
    X = jnp.concatenate((X, X_i), 1)
    key, subkey = random.split(key)
  return X / n

def _parallel(u, v):
  n = u.shape[0]
  dot_prod = u.dot(v)
  return dot_prod in [n, -n]

def _has_parallel(A, v):
  for i in range(A.shape[1]):
    if _parallel(A[:, i], v):
      return True
  return False

def onenormest(A, t, itmax):
  """Calculate the 1-norm of a matrix.
  """
  n = A.shape[0]
  AT = A.T
  X = _construct_X(n, t)
  pprint(f"X={X}")
  est = 0
  est_old = 0
  k = 1
  ind = jnp.zeros((n, 1))
  ind_hist = jnp.array([])
  S = jnp.zeros((n, t))
  zeros = jnp.zeros((n, 1))
  while True:
    Y = A @ X
    column_norms = jnp.linalg.norm(Y, ord=1, axis=0)
    max_index = jnp.argmax(column_norms)
    est = max(column_norms)
    if est > est_old:
      ind_best = max_index
      est_old = est
      w = Y[:, ind_best]

    # (1)
    if k >= 2 and est <= est_old:
      est = est_old
      pprint("(1) k >= 2 and est <= est_old")
      break
    est_old = est
    S_old = S

    if k > itmax:
      pprint("(1) k > itmax")
      break
    sign = lambda x: jnp.where(x >= 0.0, 1.0, -1.0)
    S = sign(Y)
    

    # (2)
    all_parallel = True if k > 1 else False
    def parallel_to_S_old(v):
      for i in range(t):
        if S_old[:, i].dot(v) in [n, -n]:
          return True
      return False
    for i in range(t):
      if parallel_to_S_old(S[:, i]):
        all_parallel = False
    if all_parallel:
      pprint("(2) all_parallel")
      break
    if t > 1 and k > 1:
      for i in range(t):
        tmp_S = jnp.concatenate(S_old, S[:, :i], axis=1)
        if _has_parallel(tmp_S, S[:, i]):
          new_S_i = _sample_pm_one_linearly_independent(tmp_S, n)
          while _has_parallel(tmp_S, new_S_i):
            new_S_i = _sample_pm_one_linearly_independent(tmp_S, n)
          S = S.at[:, i].set(new_S_i)
    
    # (3)
    Z = AT @ S
    pprint(f"\tZ={Z}")
    h = jnp.linalg.norm(Z, ord=jnp.inf, axis=1)

    # (4)
    if k >= 2 and max(h) == h[ind_best]:
      pprint("(4) k >= 2 and max(h) == h[ind_best]")
      break
    pprint(f"\th={h}")
    ind = jnp.flip(jnp.argsort(h))
    pprint(f"\tind={ind}")
    h = h[ind]
    pprint(f"\th={h}")

    # (5)
    if t > 1:
      contained = jnp.isin(ind[:t], ind_hist)
      if contained.all():
        pprint("(5) jnp.in1d(ind[:t], ind_hist).all()")
        break
      seen = jnp.isin(ind, ind_hist)
      pprint(f"\tseen={seen}")
      # Replace ind(1:t) by the first t indices in ind(1:n) that are not in
      # ind_hist.
      ind = jnp.concatenate((ind[~seen], ind[seen]))
    pprint(f"\tind={ind}")
    X_cols = []
    for j in range(t):
      e_ind_j = zeros.at[ind[j]].set(1.0)
      X_cols.append(e_ind_j)
    pprint(f"\tX_cols={X_cols}")
    X = jnp.concatenate((X_cols), axis=1)
    pprint(f"\tX={X}")
    # ind_hist = jnp.concatenate((int_hist, new_ind))
    new_ind = ind[:t][~jnp.isin(ind[:t], ind_hist)]
    ind_hist = jnp.concatenate((ind_hist, new_ind))

    k += 1

  v = zeros.at[ind_best].set(1.0)
  return est

if __name__ == "__main__":
  A = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  print(f"{onenormest(A, t=2, itmax=5)=}")
