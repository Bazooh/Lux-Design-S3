import jax
from functools import partial
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.state import EnvState, EnvObs, UnitState, MapTile
import jax.numpy as jnp
from enum import IntEnum
import json

class Direction(IntEnum):
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class Tiles(IntEnum):
    UNKNOWN = -1
    EMPTY = 0
    NEBULA = 1
    ASTEROID = 2

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
    # a is an int
    # 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left, 5 is sap
    flip_map = jnp.array([0, 3, 1, 4, 2, 5]) 
    return flip_map[a]

@jax.jit
def is_enemy_in_range(x_ally, y_ally, x_enemy, y_enemy, sap_range):
    """
    If enemy is out of range or unseen (-1, -1), returns False
    """

    return jnp.logical_and(
        jnp.logical_and(jnp.abs(x_ally - x_enemy) <= sap_range, 
                        jnp.abs(y_ally - y_enemy) <= sap_range),
        jnp.logical_and(x_enemy > -1, y_enemy > -1)
    )

DEFAULT_SAP_DELTAX = 0
DEFAULT_SAP_DELTAY = 0

@jax.jit
def find_delta(x,y,enemies, sap_range) :
    """
    returns a delta position of an enemy in range. if none, returns DEFAULT_SAP_DELTAX, DEFAULT_SAP_DELTAY
    """
    mask_enemy_in_range = jax.vmap(is_enemy_in_range, in_axes=(None, None, 0, 0, None))(x,y,enemies[:,0],enemies[:,1],sap_range)

    valid_indices = jnp.where(mask_enemy_in_range, size=mask_enemy_in_range.shape[0])[0]

    abs_pos_to_sap = jax.lax.select(
        mask_enemy_in_range.sum() > 0,
        jnp.take(enemies, valid_indices[0], axis=0).astype(jnp.int16),
        jnp.array([x+DEFAULT_SAP_DELTAX,y+DEFAULT_SAP_DELTAY], dtype=jnp.int16),
    )

    delta = jnp.array([abs_pos_to_sap[0]-x, abs_pos_to_sap[1]-y], dtype=jnp.int16)
    return delta

@jax.jit
def get_full_sap_action(ally_action,x,y,enemies,sap_range) :
    """
    returns a triplet(act,dx,dy), act is 0 if no enemies in range, 5 otherwwise
    """
    delta = find_delta(x, y, enemies, sap_range)
    #jax.debug.print("delta : {p}", p = delta)
    if_5_action =  jax.lax.select(
        jnp.logical_and(delta[0] == DEFAULT_SAP_DELTAX, delta[1] == DEFAULT_SAP_DELTAY),
        jnp.array([0, DEFAULT_SAP_DELTAX, DEFAULT_SAP_DELTAY], dtype=jnp.int16),
        jnp.array([5, delta[0], delta[1]], dtype=jnp.int16),
    )

    return jax.lax.select(
        ally_action == 5,
        if_5_action,
        jnp.array([ally_action, 0, 0], dtype=jnp.int16),
    )

@jax.jit
def get_action_masking_from_obs(team_id, obs: EnvObs, sap_range: int):
    """
    return the action_mask for a given team_id
    """
    enemies = obs.units.position[1- team_id]
    sap_deltas = jax.vmap(find_delta, in_axes=(0, 0, None, None))(obs.units.position[team_id, :, 0], obs.units.position[team_id, :, 1], enemies, sap_range)
    action_mask = jnp.ones((16, 6), dtype=jnp.bool_)
    can_sap_masking = (sap_deltas[:,0] != DEFAULT_SAP_DELTAX) & (sap_deltas[:,1] != DEFAULT_SAP_DELTAY)
    action_mask = action_mask.at[:, 5].set(can_sap_masking)
    return action_mask


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


def serialize_env_params(env_params: EnvParams) -> dict:
    serialized = {}
    for field_name, field_value in env_params.__dict__.items():
        if isinstance(
            field_value, (int, float, str, bool)
        ):  # Directly serializable types
            serialized[field_name] = field_value
        elif isinstance(field_value, list) or isinstance(
            field_value, tuple
        ):  # Serialize lists/tuples
            serialized[field_name] = list(field_value)
        elif isinstance(field_value, jnp.ndarray):  # Convert JAX arrays to lists
            serialized[field_name] = jax.device_get(field_value).tolist()
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

def diagonal_distances(N):
    # Create a grid of indices
    x = jnp.arange(N)
    i, j = jnp.meshgrid(x, x, indexing='ij')

    # Compute distances
    main_diag_dist = jnp.abs(i - j)
    anti_diag_dist = jnp.abs((N - 1 - i) - j)

    return main_diag_dist, anti_diag_dist

def EnvObs_from_dict(observation: dict) -> EnvObs:
    return EnvObs(
        units=UnitState(
            position=observation["units"]["position"],
            energy=observation["units"]["energy"],
        ),
        units_mask=observation["units_mask"],
        sensor_mask=observation["sensor_mask"],
        map_features=MapTile(
            energy=observation["map_features"]["energy"],
            tile_type=observation["map_features"]["tile_type"],
        ),
        relic_nodes=observation["relic_nodes"],
        relic_nodes_mask=observation["relic_nodes_mask"],
        team_points=observation["team_points"],
        team_wins=observation["team_wins"],
        steps=observation["steps"],
        match_steps=observation["match_steps"],
    )

def json_to_html(json_data: dict) -> str:
    return f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="https://s3vis.lux-ai.org/eye.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />

    <title>Lux Eye S3</title>

    <script>
window.episode = {json.dumps(json_data)};
    </script>

    <script type="module" crossorigin src="https://s3vis.lux-ai.org/index.js"></script>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
    """.strip()