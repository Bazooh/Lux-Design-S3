import dis
import jax.numpy as jnp
import jax
from regex import F

positions = jnp.array([[1,2],[1,2],[1,1],[1,2]])
mask = jnp.array([True, True, True, False])
array = jnp.zeros((4,4), dtype = jnp.int8).at[positions[:, 0], positions[:, 1]].add(mask.astype(jnp.int8))


print(array)