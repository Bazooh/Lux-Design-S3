import jax
from functools import partial
from luxai_s3.params import EnvParams
from luxai_s3.params import env_params_ranges
import jax.numpy as jnp

@jax.jit
def mirror_grid(array):
    """
    Input: (H, W)
    Output: (H, W)
    """
    return jnp.flip(jnp.transpose(array))

@partial(jax.jit, static_argnums=(0,))
def symmetrize(team_id, array):
    my_part = jax.lax.cond(
        team_id == 0, 
        lambda: jnp.fliplr(jnp.triu(jnp.fliplr(array))), 
        lambda: jnp.fliplr(jnp.tril(jnp.fliplr(array))),
    )
    symmetric_my_part = mirror_grid(my_part)
    return  symmetric_my_part + my_part - jnp.fliplr(jnp.diag(jnp.diag(jnp.fliplr(my_part))))

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

def mirror_relic_positions_arrays(relic_positions):
    """
    Input: (6, 2) of obs.relic_positions (or memory.relics_found_positions)
    Output: (6, 2) of obs.relic_positions  (or memory.relics_found_positions)
    """
    relic_pos = relic_positions.reshape((2, 3, 2))
    mirrored_relic_pos = jax.vmap(jax.vmap(mirror_position))(relic_pos)
    empty = -100 * jnp.ones((3,2), dtype = jnp.int32)
    relic_positions = jnp.stack([
        jnp.where(relic_pos[0] > 0, relic_pos[0], jnp.where(relic_pos[1] > 0, mirrored_relic_pos[1], empty)), 
        jnp.where(relic_pos[1] > 0, relic_pos[1], jnp.where(relic_pos[0] > 0, mirrored_relic_pos[0], empty))
    ]
    )
    return relic_positions.reshape((6, 2))

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

def manhattan_distance_to_nearest_point(source_pos, n):
    """    
    Args:
        source_pos (k, 2): position of the source points
        n: grid size
    Returns:
        distances (n,n): Matrix of Manhattan distances to nearest source
    """
    @jax.jit
    def compute_min_distance(r, c):
        return jnp.min(jnp.abs(source_pos[:, 0] - r) + jnp.abs(source_pos[:, 1] - c)) # Compute minimum distance to any point with value 1
    
    row_indices, col_indices = jnp.indices((n,n))
    distances = jax.vmap(jax.vmap(compute_min_distance))(row_indices, col_indices)
    
    return jnp.clip(distances, 0, n//2)