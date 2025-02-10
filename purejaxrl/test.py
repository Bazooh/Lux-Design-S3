from utils import CustomTrainState

import jax
import jax.numpy as jnp

state = CustomTrainState.create(
    apply_fn=None, params={}, tx=None, opt_state=None, batch_stats={"mean": jnp.array(0.0)}
)

print(jax.tree_util.tree_structure(state))  # Should not raise an error