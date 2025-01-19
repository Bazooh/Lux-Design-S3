import jax.numpy as jnp
import jax
obs_sensor_mask = jnp.zeros((24,24), dtype = jnp.bool)
last_obs_sensor_mask = jnp.ones((24,24), dtype = jnp.bool)
x = jax.numpy.mean(jax.numpy.clip(obs_sensor_mask - last_obs_sensor_mask, min=0, max=1))
print(x)