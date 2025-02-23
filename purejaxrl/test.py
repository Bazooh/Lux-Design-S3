import jax.numpy as jnp
import jax
from utils import binary_cross_entropy
arr_true = jax.random.uniform(jax.random.PRNGKey(0), (4, 4)) > 0.5
arr_true_v = jnp.stack([1 - arr_true, arr_true])
arr_pred = jax.random.uniform(jax.random.PRNGKey(0), (4, 4))
arr_pred_v = jnp.stack([1 - arr_pred, arr_pred])
error = jax.vmap(jax.vmap(jax.vmap(binary_cross_entropy, in_axes=(0, 0)), in_axes=(0, 0)), in_axes=(0, 0))(arr_true_v, arr_pred_v).mean()
print(error)