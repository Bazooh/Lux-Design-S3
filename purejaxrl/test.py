import jax.numpy as jnp
import jax
@jax.jit
def sum_of_positive(x):
  return jnp.where(x > 0, x, 0)

print(sum_of_positive(jnp.array([-1, 2, 3, -4])))

