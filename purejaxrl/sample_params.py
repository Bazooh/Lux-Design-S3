import jax
import jax.numpy as jnp
from luxai_s3.params import EnvParams
from luxai_s3.params import env_params_ranges

@jax.vmap
def sample_params_fn(rng_key):
    randomized_game_params = dict()
    for k, v in env_params_ranges.items():
        rng_key, subkey = jax.random.split(rng_key)
        if isinstance(v[0], int):
            randomized_game_params[k] = jax.random.choice(
                subkey, jax.numpy.array(v, dtype=jnp.int16)
            )
        else:
            randomized_game_params[k] = jax.random.choice(
                subkey, jax.numpy.array(v, dtype=jnp.float32)
            )
    params = EnvParams(**randomized_game_params)
    return params