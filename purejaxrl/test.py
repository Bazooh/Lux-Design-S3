import jax.numpy as jnp
import jax
p = jnp.array([[0,1],[2,3]])
def sym(pos):
    print(23*jnp.ones(2) - jnp.flip(pos))

print(jax.vmap(sym)(p))