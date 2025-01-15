import jax.numpy as jnp
import jax

image = jnp.eye(4)
mask = jnp.array([
	[0, 0, 0, 0],
	[0, 1, 1, 0],
	[0, 1, 1, 0],
	[0, 0, 0, 0]
	
])
print(jnp.multiply(image == 1, mask))