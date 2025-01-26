from functools import partial
import jax, chex
import jax.numpy as jnp
from luxai_s3.params import EnvParams
from luxai_s3.params import env_params_ranges
import numpy as np
import termplotlib as tpl
import flax
def sample_params(rng_key, match_count_per_episode = 5):
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
    params = EnvParams(match_count_per_episode = match_count_per_episode, **randomized_game_params)
    return params

@jax.jit
def sample_group_action(key, logits_group, action_temperature: float = 1.0):
    """
    key: PRNG key for sampling.
    logits_group: Logits for the action of a group of ships. Shape: (16, action_dim).
    """
    def sample_action(key: chex.PRNGKey, logits: chex.Array, action_temperature: float = 1.0):
        """
        key: PRNG key for sampling.
        logits: Logits for the action of a single ship. Shape: (action_dim).
        """
        scaled_logits = (logits - jnp.mean(logits, axis=-1, keepdims=True)) / action_temperature
        action = jax.random.categorical(key=key, logits=scaled_logits, axis=-1)
        return action
    
    # Split the PRNG key into one key per group element
    group_keys = jax.random.split(key, num=logits_group.shape[0])
    
    # Use vmap to vectorize the sampling over group_keys and logits_group
    action_group = jax.vmap(sample_action, in_axes=(0, 0, None))(group_keys, logits_group, action_temperature)
    
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

def init_network_params(key, network, init_x, print_summary = False):
    init_x = {feat: jnp.expand_dims(value, axis=0) for feat, value in init_x.items()}
    network_params = network.init(key, **init_x)["params"]
    
    if print_summary: 
        tabulate_fn = flax.linen.tabulate(network, jax.random.key(0), compute_flops=True, compute_vjp_flops=True)
        print(tabulate_fn(**init_x))
         
    return network_params

def serialize_metadata(metadata: dict) -> dict:
    serialized = {}
    for key, value in metadata.items():
        if isinstance(value, (int, float, str, bool)):  # Directly serializable types
            serialized[key] = value
        elif isinstance(value, jax.Array) or isinstance(value, jax.numpy.ndarray):  # JAX or NumPy arrays
            serialized[key] = jax.device_get(value).tolist()
        elif isinstance(value, dict):  # Nested dictionaries
            serialized[key] = serialize_metadata(value)
        elif isinstance(value, (list, tuple)):  # Lists or tuples
            serialized[key] = [serialize_metadata(v) if isinstance(v, dict) else v for v in value]
    return serialized

@jax.jit
def mirror_grid(array):
    """
    Input: (H, W)
    Output: (H, W)
    """
    return jnp.flip(jnp.transpose(array))

@jax.jit
def diagonal_of_array(array):
    return jnp.fliplr(jnp.diag(jnp.diag(jnp.fliplr(array))))

@partial(jax.jit, static_argnums=(0,))
def symmetrize(team_id, array):
    my_part = jax.lax.cond(
        team_id == 0, 
        lambda: jnp.fliplr(jnp.triu(jnp.fliplr(array))), 
        lambda: jnp.fliplr(jnp.tril(jnp.fliplr(array))),
    )
    symmetric_my_part = mirror_grid(my_part)
    return  symmetric_my_part + my_part - diagonal_of_array(my_part)


@jax.jit
def mirror_position(pos):
    """
    Input: Shape (2): (x,y)
    Output: Shape 2: (23-y, 23-x)
    """
    return 23*jnp.ones(2, dtype=int) - jnp.flip(pos)


@jax.jit
def mirror_action(a):
    # a is (3,)
    # 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left, 5 is sap
    flip_map = jnp.array([0, 3, 1, 4, 2]) 
    @jax.jit
    def flip_move_action(a):
        # a is (3,)
        # 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left
        a = a.at[0].set(flip_map[a[0]])  # Map the first element using flip_map
        return a

    @jax.jit
    def flip_sap_action(a):
        # a = (5, x, y), (x,y) should be replaced by (-y, -x)
        a = a.at[1:].set(jnp.array([-1*a[2], -1*a[1]]))
        return a
    
    a = jax.lax.cond(
        a[0] < 5,
        flip_move_action,
        flip_sap_action,
        a,
    )
    return a


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