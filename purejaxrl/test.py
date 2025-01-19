import jax.numpy as jnp
import jax
N = 2
image = jnp.zeros((N, 24, 24, 3), dtype=jnp.float32)
vector = jnp.zeros((N, 32), dtype=jnp.float32)

# Transpose image for channel-last format
B, H, W, C = image.shape

# Process the vector
vector = vector.reshape((B, 1, 1, -1))  # (N, V) -> (N, 1, 1, V)
vector = jnp.tile(vector, (1, H, W, 1))  # (N, 1, 1, V) -> (N, H, W, V)
# Concatenate
x = jnp.concatenate((image, vector), axis=-1)  # (N, H, W, C + V)
print(x.shape)