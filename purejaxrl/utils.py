import jax
import jax.numpy as jnp
from luxai_s3.params import EnvParams
from luxai_s3.params import env_params_ranges

def sample_params(rng_key):
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

@jax.jit
def sample_action(key, logits):
    action = jax.random.categorical(key=key, logits=logits, axis=-1)  # Shape: (N, 16)
    return action

@jax.jit
def sample_greedy_action(logits): # input (N, 16, 6)
    action = jax.numpy.argmax(logits, axis=-1)  # Shape: (N, 16)
    return action

@jax.jit
def get_logprob(logits, mask_awake, action):
    log_prob_group = jax.nn.log_softmax(logits, axis=-1)  # Shape: (N, 16, 6)
    log_prob_a = jnp.take_along_axis(log_prob_group, action[..., None], axis=-1).squeeze(axis=-1)  # Shape: (N, 16)
    log_prob_a_masked = log_prob_a * mask_awake  # Shape: (N, 16)
    log_prob= jnp.mean(log_prob_a_masked, axis=-1)/ jnp.sum(mask_awake, axis=-1)  # Shape: (N,)
    return(log_prob)

@jax.jit
def get_entropy(logits):
    log_prob_group = jax.nn.log_softmax(logits, axis=-1)  # Shape: (N, 16, 6)
    entropy = -jnp.mean(jnp.sum(jnp.exp(log_prob_group) * log_prob_group, axis=-1), axis=-1)
    return(entropy)


def get_obs_batch(obs, player_list):
    return [{key: obs[player][key] for key in obs[player]} for player in player_list]

def init_network_params(key, network, init_x):
    init_x = {feat: jnp.expand_dims(value, axis=0) for feat, value in init_x.items()}
    network_params = network.init(key, **init_x)
    return network_params