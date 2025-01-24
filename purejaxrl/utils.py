import jax
import jax.numpy as jnp
from luxai_s3.params import EnvParams
from luxai_s3.params import env_params_ranges

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
def sample_action(key, logits, noise_std=0.0):
    """
    Samples an action with optional noise added to logits for exploration.

    Args:
        key: PRNG key for sampling.
        logits: Logits output from the policy network. Shape: (N, 16, action_dim).
        noise_std: Standard deviation of Gaussian noise to add.

    Returns:
        action: Sampled action. Shape: (N, 16).
    """
    # Add Gaussian noise to logits
    noise = jax.random.normal(key, shape=logits.shape) * noise_std
    noisy_logits = logits + noise

    # Sample action from noisy logits
    action = jax.random.categorical(key=key, logits=noisy_logits, axis=-1)  # Shape: (N, 16)
    return action

@jax.jit
def sample_greedy_action(logits): # input (N, 16, 6)
    action = jax.numpy.argmax(logits, axis=-1)  # Shape: (N, 16)
    return action

@jax.jit
def get_logprob(logits, mask_awake, action):
    log_prob_group = jax.nn.log_softmax(logits, axis=-1)  # Shape: (N, 16, 6)
    log_prob_a = jnp.take_along_axis(log_prob_group, action[..., None], axis=-1).squeeze(axis=-1)  # Shape: (N, 16)
    log_prob_a_masked = jnp.where(mask_awake, log_prob_a, 0.0)  # Mask invalid positions
    log_prob = jnp.sum(log_prob_a_masked, axis=-1)  # Shape: (N,)
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


def mirror_grid(array):
    """
    Input: (H, W)
    Output: (H, W)
    """
    return jnp.flip(jnp.transpose(array))


def mirror_position(pos):
    """
    Input: Shape (2): (x,y)
    Output: Shape 2: (23-y, 23-x)
    """
    return 23*jnp.ones(2, dtype=int) - jnp.flip(pos)


def mirror_action(a):
    # a is (3,)
    # 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left, 5 is sap
    flip_map = jnp.array([0, 3, 4, 1, 2]) 
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
