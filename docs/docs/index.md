# Welcome to MkDocs

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

# One-Norm Estimation

## Getting it to JIT
A big reason to use JAX over numpy and scipy is for performance.
These performance gains are achieved with just-in-time compilation.
Ideally, the standard library of JAX is fully JITtable.
Thus, it would be desirable for onenormest to jit.

We highlight the sections of our current implementation that currently won't JIT and explain why they don't JIT.

```py
```

