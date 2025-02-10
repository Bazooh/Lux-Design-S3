import jax, chex
import jax.numpy as jnp
import numpy as np
import flax
import orbax.checkpoint
from flax.training import orbax_utils
from typing import Any
import os
from functools import partial

@partial(jax.jit, static_argnums=(3))
def sample_group_action(key, logits_group: chex.Array, action_mask: chex.Array, action_temperature: float = 1.0):
    """
    key: PRNG key for sampling.
    logits_group: Logits for the action of a group of ships. Shape: (16, action_dim).
    action_mask: Mask for the action of a group of ships. Shape: (16, action_dim).
    """
    @partial(jax.jit, static_argnums=(3))
    def sample_action(key: chex.PRNGKey, logits: chex.Array, action_mask: chex.Array, action_temperature: float = 1.0):
        """
        key: PRNG key for sampling.
        logits: Logits for the action of a single ship. Shape: (action_dim).
        action_mask: Mask for the action of a single ship. Shape: (action_dim).
        """
        scaled_logits = (logits - jnp.mean(logits, axis=-1, keepdims=True)) / action_temperature
        scaled_logits = jnp.where(action_mask, scaled_logits, -1e9)  # Mask invalid
        action = jax.random.categorical(key=key, logits=scaled_logits, axis=-1)
        return action
    
    # Split the PRNG key into one key per group element
    group_keys = jax.random.split(key, num=logits_group.shape[0])
    
    # Use vmap to vectorize the sampling over group_keys and logits_group
    action_group = jax.vmap(sample_action, in_axes=(0, 0, 0, None))(group_keys, logits_group, action_mask, action_temperature)
    
    return action_group

@jax.jit
def get_logprob(logits, mask_awake, action):
    log_prob_group = jax.nn.log_softmax(logits, axis=-1)  # Shape: (N, 16, 6)
    log_prob_a = jnp.take_along_axis(log_prob_group, action[..., None], axis=-1).squeeze(axis=-1)  # Shape: (N, 16)
    log_prob_a_masked = jnp.where(mask_awake, log_prob_a, 0.0)  # Mask invalid positions
    log_prob = jnp.sum(log_prob_a_masked, axis=-1)  # Shape: (N,)
    return(log_prob)

@jax.jit
def get_entropy(logits, mask_awake):
    """
    logits: Array of shape (16, action_dim), where N is the number of rows.
    mask_awake: Boolean mask of shape (16,), indicating which rows to include.
    """
    def entropy_row(logits_row):
        probs = jax.nn.softmax(logits_row)
        return -jnp.sum(probs * jnp.log(probs + 1e-9))  
    
    entropies = jax.vmap(entropy_row)(logits)
    return jnp.mean(jnp.where(mask_awake, entropies, 0.0))



def get_obs_batch(obs, player_list):
    return [{key: obs[player][key] for key in obs[player]} for player in player_list]

from flax.training.train_state import TrainState
class CustomTrainState(TrainState):
    batch_stats: Any
    
def init_state_dict(key, model, init_x: Any, print_summary = False):
    init_x = {feat: jnp.expand_dims(value, axis=0) for feat, value in init_x.items()}
    variables = model.init(key, **init_x, train=False)
    if print_summary: 
        tabulate_fn = flax.linen.tabulate(model, jax.random.key(0), compute_flops=False, compute_vjp_flops=False)
        print(tabulate_fn(**init_x))
    return(variables)


def plot_stats(stats_arrays):
    """
    Plot stats for both players over time.

    Parameters
    ----------
    stats_arrays : dict of {str: dict of {str: np.ndarray}}
        A dictionary with two keys: "episode_stats_player_0" and "episode_stats_player_1".
        Each of these keys contains a dictionary with the following keys: "episode_return", "episode_length", "episode_points", "episode_wins", etc .
        Each of these keys contains a numpy array of length 120, containing the respective stat at each of the 120 episodes.
    -------
    """
    import termplotlib as tpl
    stat_names = list(stats_arrays["episode_stats_player_0"].keys())
    stat_names.reverse()
    for stat in stat_names:
        y0 = stats_arrays["episode_stats_player_0"][stat]
        y1 = stats_arrays["episode_stats_player_1"][stat]
        x = np.arange(len(y0))
        
        fig = tpl.figure()
        fig.plot(x, y0, label=f"{stat} for player 0", width = 100, height = 15) 
        fig.plot(x, y1, label=f"{stat} for player 1", width = 100, height = 15) 
        fig.show()
    
    print("-" * 45)
    stat_names.reverse()
    print(f"{'Stat Name':<20} {'Player 0':>10} | {'Player 1':>10}")
    print("-" * 45)
    for stat in stat_names:
        y0 = stats_arrays["episode_stats_player_0"][stat]
        y1 = stats_arrays["episode_stats_player_1"][stat]
        print(f"{stat:<20} {y0[-1]:>10.2f} | {y1[-1]:>10.2f}")

from luxai_s3.env import EnvObs
from typing import Any
def EnvObs_to_dict(obs: EnvObs) ->  dict[str, Any]:
    return {
        "units": {
            "position": obs.units.position,
            "energy": obs.units.energy,
        },
        "units_mask": obs.units_mask,
        "sensor_mask": obs.sensor_mask,
        "map_features": {
            "energy": obs.map_features.energy,
            "tile_type": obs.map_features.tile_type,
        },
        "relic_nodes": obs.relic_nodes,
        "relic_nodes_mask": obs.relic_nodes_mask,
        "team_points": obs.team_points,
        "team_wins": obs.team_wins,
        "steps": obs.steps,
        "match_steps": obs.match_steps
    }

def create_checkpoint_manager(path):
    return orbax.checkpoint.CheckpointManager(
        directory = path,
        checkpointers = orbax.checkpoint.PyTreeCheckpointer(), 
        options = orbax.checkpoint.CheckpointManagerOptions(create=True)
    )

def save_state_dict(state_dict, checkpoint_manager, step=0):
    """Saves the model state using Orbax."""
    save_args = orbax_utils.save_args_from_target(state_dict)
    checkpoint_manager.save(step, state_dict, save_kwargs={'save_args': save_args})

def restore_state_dict(path, step=None):
    """Restores the model state using Orbax."""
    checkpoint_manager = orbax.checkpoint.CheckpointManager(path, orbax.checkpoint.PyTreeCheckpointer())
    if step is None:
        step = checkpoint_manager.latest_step()
    restored_state_dict = checkpoint_manager.restore(step)
    return restored_state_dict

def restore_state_dict_cpu(path, step=None):
    """Restores the model state using Orbax but forces CPU-compatible sharding."""
    checkpoint_manager = orbax.checkpoint.CheckpointManager(path, orbax.checkpoint.PyTreeCheckpointer())
    if step is None:
        step = checkpoint_manager.latest_step()
    structure = checkpoint_manager.item_metadata(step)
    
    restored_state_dict = checkpoint_manager.restore(
        step,
        restore_kwargs={
            "restore_args": jax.tree_map(
                lambda _: orbax.checkpoint.RestoreArgs(restore_type=np.ndarray), structure
            )
        },
    )
    return restored_state_dict