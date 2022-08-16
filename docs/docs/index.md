# Welcome to MkDocs

# One-Norm Estimation

## Getting it to JIT
A big reason to use JAX over numpy and scipy is for performance.
These performance gains are achieved with just-in-time compilation.
Ideally, the standard library of JAX is fully JITtable.
Thus, it would be desirable for onenormest to jit.

We highlight the sections of our current implementation that currently won't JIT and explain why they don't JIT.

```py
```

# Procedure for JITting a function
Given a function `f` that you want to jit, we give a general procedure here on how to JIT it.

## Step 1: `jit` it

```py
from jax import jit

jit(f)
```

## Step 2: Observe the error
If the JIT fails then you will receive an error.
The stack trace should also include a link to JAX documentation which explains the failure.
```
ConcretizationTypeError                   Traceback (most recent call last)
/usr/local/lib/python3.7/dist-packages/jax/_src/numpy/lax_numpy.py in arange(start, stop, step, dtype)
   2105   else:
   2106     start = require(start, msg("start"))
-> 2107     stop = None if stop is None else require(stop, msg("stop"))
   2108     step = None if step is None else require(step, msg("step"))
   2109     if step is None and start == 0 and stop is not None:

ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>
It arose in jax.numpy.arange argument `stop`.
While tracing the function onenormest at <ipython-input-2-a95eb20fecee>:46 for jit, this concrete value was not available in Python because it depends on the value of the argument 't'.

See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
```

## Step 3: Fix the error
Make the minimal change needed and take note of its diff.
```diff
+ fix example
- bad example
```

## Step 4: Repeat
Continue iterating until calling `jit(f)` no longer gives an error.
