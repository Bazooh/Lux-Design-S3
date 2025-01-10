import jax
import jax.numpy as jnp
from functools import partial

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