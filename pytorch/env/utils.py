
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.state import EnvState, EnvObs, UnitState, MapTile
import numpy as np
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

def mirror_grid(array):
    """
    Input: (H, W)
    Output: (H, W)
    """
    return np.flip(np.transpose(array))

def symmetrize(team_id, array):
    if team_id == 0:
        my_part = np.fliplr(np.triu(np.fliplr(array)))
    else: 
        my_part = np.fliplr(np.tril(np.fliplr(array)))
    symmetric_my_part = mirror_grid(my_part)
    return  symmetric_my_part + my_part - np.fliplr(np.diag(np.diag(np.fliplr(my_part))))

def mirror_position(pos):
    """
    Input: Shape (2): (x,y)
    Output: Shape 2: (23-y, 23-x)
    """
    return 23*np.ones(2, dtype=int) - np.flip(pos)

def mirror_action(a):
    # a is an int
    # 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left, 5 is sap
    flip_map = np.array([0, 3, 1, 4, 2, 5]) 
    return flip_map[a]

def is_enemy_in_range(x_ally, y_ally, x_enemy, y_enemy, sap_range):
    """
    If enemy is out of range or unseen (-1, -1), returns False
    """

    return np.logical_and(
        np.logical_and(np.abs(x_ally - x_enemy) <= sap_range, 
                        np.abs(y_ally - y_enemy) <= sap_range),
        np.logical_and(x_enemy > -1, y_enemy > -1)
    )

DEFAULT_SAP_DELTAX = 0
DEFAULT_SAP_DELTAY = 0

def find_delta(x,y,enemies, sap_range) :
    """
    returns a delta position of an enemy in range. if none, returns DEFAULT_SAP_DELTAX, DEFAULT_SAP_DELTAY
    """
    mask_enemy_in_range = np.array([
        is_enemy_in_range(x,y,enemies[i,0],enemies[:,1],sap_range)
        for i in range(16)
    ])

    valid_indices = np.where(mask_enemy_in_range)[0]

    if mask_enemy_in_range.sum() > 0:
        abs_pos_to_sap = np.take(enemies, valid_indices[0], axis=0).astype(np.int16)
    else:
        abs_pos_to_sap = np.array([x+DEFAULT_SAP_DELTAX,y+DEFAULT_SAP_DELTAY], dtype=np.int16)


    delta = np.array([abs_pos_to_sap[0]-x, abs_pos_to_sap[1]-y], dtype=np.int16)
    return delta

def get_full_sap_action(ally_action,x,y,enemies,sap_range) :
    """
    returns a triplet(act,dx,dy), act is 0 if no enemies in range, 5 otherwwise
    """
    delta = find_delta(x, y, enemies, sap_range)
    #jax.debug.print("delta : {p}", p = delta)
    if np.logical_and(delta[0] == DEFAULT_SAP_DELTAX, delta[1] == DEFAULT_SAP_DELTAY):
        if_5_action = np.array([0, DEFAULT_SAP_DELTAX, DEFAULT_SAP_DELTAY], dtype=np.int16),
    else:
        if_5_action = np.array([5, delta[0], delta[1]], dtype=np.int16),

    if ally_action == 5:
        return if_5_action
    else:
        np.array([ally_action, 0, 0], dtype=np.int16)


def get_action_masking_from_obs(team_id, obs: EnvObs, sap_range: int):
    """
    return the action_mask for a given team_id
    """
    enemies = obs["units"]["position"][1- team_id]
    sap_deltas = np.array([
        find_delta(obs["units"]["position"][team_id, i, 0], obs["units"]["position"][team_id, i, 1], enemies, sap_range)
        for i in range(16)
    ])
    action_mask = np.ones((16, 6), dtype=np.bool_)
    can_sap_masking = (sap_deltas[:,0] != DEFAULT_SAP_DELTAX) & (sap_deltas[:,1] != DEFAULT_SAP_DELTAY)
    action_mask[:, 5] = can_sap_masking
    return action_mask


def mirror_relic_positions_arrays(relic_positions):
    """
    Input: (6, 2) of obs.relic_positions (or memory.relics_found_positions)
    Output: (6, 2) of obs.relic_positions  (or memory.relics_found_positions)
    """
    relic_pos = relic_positions.reshape((2, 3, 2))
    mirrored_relic_pos = np.array([[mirror_position(relic_pos[i,j]) for j in range(3)] for i in range(2)])
    empty = -100 * np.ones((3,2), dtype = np.int32)
    relic_positions = np.stack([
        np.where(relic_pos[0] > 0, relic_pos[0], np.where(relic_pos[1] > 0, mirrored_relic_pos[1], empty)), 
        np.where(relic_pos[1] > 0, relic_pos[1], np.where(relic_pos[0] > 0, mirrored_relic_pos[0], empty))
    ]
    )
    return relic_positions.reshape((6, 2))


def manhattan_distance_to_nearest_point(source_pos, n):
    """    
    Args:
        source_pos (k, 2): position of the source points
        n: grid size
    Returns:
        distances (n,n): Matrix of Manhattan distances to nearest source
    """

    def compute_min_distance(r, c):
        return np.min(np.abs(source_pos[:, 0] - r) + np.abs(source_pos[:, 1] - c)) # Compute minimum distance to any point with value 1
    
    distances = np.array([[compute_min_distance(x,y) for y in range(n)] for x in range(n)])
    
    return np.clip(distances, 0, n//2)


def diagonal_distances(N):
    # Create a grid of indices
    x = np.arange(N)
    i, j = np.meshgrid(x, x, indexing='ij')

    # Compute distances to diagonals
    main_diag_dist = np.abs(i - j)
    anti_diag_dist = np.abs((N - 1 - i) - j)
    center = (N - 1) / 2  # Center coordinates (can be fractional for even N)
    dist_to_center = np.abs(i - center) + np.abs(j - center)
    dist_to_top = i
    dist_to_left = j

    return main_diag_dist, anti_diag_dist, dist_to_center, dist_to_top, dist_to_left


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