import jax.numpy as jnp
import jax

array = jnp.arange(16).reshape(4, 4) 
pos = jnp.array([[0, 1],
                 [1, 1],
                 [2, 1],
                 [3, 1]])
mask = jnp.array([True, False, True, True],)
units_out_of_range_image = jnp.zeros((4, 4), dtype = jnp.int8).at[pos[:, 0], pos[:, 1]].set(mask)


print(units_out_of_range_image)