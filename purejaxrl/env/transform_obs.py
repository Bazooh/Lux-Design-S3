from abc import ABC, abstractmethod

import gymnax.environments.spaces
from src.luxai_s3.state import EnvObs
import jax
from functools import partial
from luxai_s3.params import EnvParams, env_params_ranges
import gymnax
from typing import Any
import jax.numpy as jnp
import numpy as np
from purejaxrl.env.utils import (
    mirror_grid, 
    mirror_position, 
    symmetrize, 
    manhattan_distance_to_nearest_point, 
    diagonal_distances, 
    Tiles, 
    get_action_masking_from_obs
)
from purejaxrl.env.memory import RelicPointMemoryState

class TransformObs(ABC):
    """
    Abstract base class for converting observations into tensor representations.
    """

    def __init__(self):
        pass


    @abstractmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def convert(
        self,
        team_id: int,
        obs: EnvObs,
        params: EnvParams,
        memory_state: Any,
    ):
        """
        Convert an observation into an np.array representation.

        Args:
            team_id_str: The id of the team for which we are computing the observation.
            obs: The environment observation.
            params: The environment parameters.
            reward: The reward for the current step. If None, the reward is not computed.

        Returns:
            A tensor representation of the observation.
        """
        pass
        
class HybridTransformObs(TransformObs):

    """
    This feature extractor is hybrid.
    It returns a dictionnay with the following:

        'image': Global visual features: The image representation of the map. Shape is (N_channels, 24, 24)
        'vector': Vectors features: A feature vector for each agent. Shape is (16, N_v). Each vector contains 
            - game parameters 
            - current team points
            - relic nodes relative positions
            - agent energy
    """
    
    def __init__(self):
        super().__init__()
        params = EnvParams()
        self.image_features = { # Key: Name of the feature, Value: Number of channels required representing the feature
            # Units Related
            "Unknown": 1,
            "Asteroid": 1,
            "Nebula": 1,
            "Energy_Field": 1,
            # Units Related
            "Ally_Units_Count": 1,
            "Ally_Units_Energy": 1,
            "Enemy_Units_Count": 1,
            "Enemy_Units_Energy": 1,
            # Memory Related
            "Relic": 1,
            "Distance_to_relic": 1,
            "Relic_Circle": 1,
            "Points": 1,
            "Last Visit": 1,
            # Other
            "Distance to frontier": 1,
            "Distance to main diagonal": 1,
            "Distance to center": 1,
            "Xcoord": 1,
            "Ycoord": 1,
        }
        self.vector_features = { # Key: Name of the feature, Value: Size of the vector representing the feature
            # 11 Game Parameters
            "unit_move_cost":  1, # 1 float
            "unit_sensor_range": 1, # 1 float
            "nebula_tile_vision_reduction": 1, # 1 float
            "nebula_tile_energy_reduction": 1, # 1 float
            "unit_sap_cost": 1, # 1 float
            "unit_sap_range": 1, # 1 float
            "unit_sap_dropoff_factor": 1, # 1 float
            "unit_energy_void_factor": 1, # 1 float
            "nebula_tile_drift_speed": 1, # 1 float
            "energy_node_drift_speed": 1, # 1 float
            "energy_node_drift_magnitude": 1, # 1 float
            # 6 Game State
            "team_points": 1, # 1 float
            "opponent_points": 1, # 1 float
            "points_gained": 1, # 1 float
            "relic_found_1": 1, # 1 bool
            "relic_found_2": 1, # 1 bool
            "relic_found_3": 1, # 1 bool
        }
        self.vector_size = sum(self.vector_features.values())
        self.image_channels = sum(self.image_features.values())

        self.vector_mean = { key : np.mean(env_params_ranges[key]) if key in env_params_ranges else 0 for key in self.vector_features.keys()}
        self.vector_std = { key : np.std(env_params_ranges[key]) if key in env_params_ranges else 1 for key in self.vector_features.keys()}
        self.time_discretization = 5
        self.time_size = (params.max_steps_in_match + 1)//self.time_discretization + (params.match_count_per_episode + 1)
        self.vector_std_values = jnp.array(list(self.vector_std.values()))
        self.vector_mean_values = jnp.array(list(self.vector_mean.values()))
        self.observation_space = gymnax.environments.spaces.Dict({
                            'image': gymnax.environments.spaces.Box(low=-1, high=1, shape=(self.image_channels, 24, 24), dtype=jnp.int8), 
                            'vector': gymnax.environments.spaces.Box(low=-1, high=1, shape=(self.vector_size), dtype=jnp.float32),
                            'time': gymnax.environments.spaces.Box(low=0, high=1, shape=(self.time_size), dtype=jnp.float32),
                            'position': gymnax.environments.spaces.Box(low=0, high=23, shape=(16, 2), dtype=jnp.int8), 
                            'mask_awake': gymnax.environments.spaces.Box(low=0, high=1, shape=(16), dtype=jnp.int8),
                            'action_mask': gymnax.environments.spaces.Box(low=0, high=1, shape=(16, 6), dtype=jnp.int8),
        })
    @partial(jax.jit, static_argnums=(0, 1))
    def convert(
        self, 
        team_id: int,
        obs: EnvObs, 
        memory_state: RelicPointMemoryState,
        params: EnvParams,
    ):
        image = jnp.zeros(
            (self.image_channels, obs.map_features.energy.shape[0], obs.map_features.energy.shape[1]),
            dtype=jnp.float32,
        ) 

        vector = jnp.zeros(
            (self.vector_size), 
            dtype=jnp.float32
        )
    
        ############# HANDLES IMAGE ##############
        image = image.at[0].set(obs.sensor_mask) # unknown
        image = image.at[1].set(symmetrize(team_id,(obs.map_features.tile_type == Tiles.ASTEROID).astype(jnp.int8))) # asteroid
        image = image.at[2].set(symmetrize(team_id,(obs.map_features.tile_type == Tiles.NEBULA).astype(jnp.int8))) # nebula
        image = image.at[3].set(symmetrize(team_id,(obs.map_features.energy / 20).astype(jnp.float32))) # energy field

        # enemy units and ally units
        positions = jnp.array(obs.units.position)
        image = image.at[
            4,
            positions[team_id, :, 0],
            positions[team_id, :, 1],
        ].add(obs.units_mask[team_id])

        image = image.at[
            5,
            positions[team_id, :, 0],
            positions[team_id, :, 1],
        ].add(jnp.maximum(1.0, obs.units.energy[team_id] + 1) / 400)
        
        image = image.at[
            6,
            positions[1-team_id, :, 0],
            positions[1-team_id, :, 1],
        ].add(obs.units_mask[1-team_id])

        image = image.at[
            7,
            positions[1- team_id, :, 0],
            positions[1- team_id, :, 1],
        ].add(jnp.maximum(1.0, obs.units.energy[1 - team_id] + 1) / 400)

        image = image.at[8].set(memory_state.relics_found_image)
        distance_matrix = symmetrize(team_id, manhattan_distance_to_nearest_point(source_pos=memory_state.relics_found_positions, n=24))
        image = image.at[9].set(distance_matrix / 24)
        relic_circle = jnp.clip(jax.scipy.signal.convolve2d(memory_state.relics_found_image == 1, jnp.ones((5, 5)), mode='same'), min=0, max=1)
        image = image.at[10].set(relic_circle)
        image = image.at[11].set(memory_state.points_found_image)
        image = image.at[12].set(memory_state.last_visits_timestep / (obs.steps + 1))
        
        d1, d2, d3, d4, d5 = diagonal_distances(24)
        image = image.at[13].set(d1 / 23)
        image = image.at[14].set(d2 / 23)
        image = image.at[15].set(d3 / 23)
        image = image.at[16].set(symmetrize(team_id, mirror_grid(d4)) / 23 if team_id == 1 else symmetrize(team_id,d4) / 23)
        image = image.at[17].set(symmetrize(team_id, mirror_grid(d5)) / 23 if team_id == 1 else symmetrize(team_id,d5) / 23)

        ############# HANDLES VECTOR ##############
        vector = vector.at[0].set(params.unit_move_cost)
        vector = vector.at[1].set(params.unit_sensor_range)
        vector = vector.at[2].set(params.nebula_tile_vision_reduction)
        vector = vector.at[3].set(params.nebula_tile_energy_reduction)
        vector = vector.at[4].set(params.unit_sap_cost)
        vector = vector.at[5].set(params.unit_sap_range)
        vector = vector.at[6].set(params.unit_sap_dropoff_factor)
        vector = vector.at[7].set(params.unit_energy_void_factor)
        vector = vector.at[8].set(params.nebula_tile_drift_speed)
        vector = vector.at[9].set(params.energy_node_drift_speed)
        vector = vector.at[10].set(params.energy_node_drift_magnitude)

        vector = vector.at[11].set(obs.team_points[team_id])
        vector = vector.at[12].set(obs.team_points[1-team_id])
        vector = vector.at[13].set(memory_state.points_gained)
        vector = vector.at[14:17].set(memory_state.relics_found_mask[0:3])
        rescaled_vector = (vector - self.vector_mean_values) / self.vector_std_values
        
        ############# HANDLES TIME WITH OHE ##############
        current_step = obs.steps % (params.max_steps_in_match + 1)
        current_match = obs.steps // (params.max_steps_in_match + 1)

        time_vector = jnp.zeros((self.time_size), dtype=jnp.float32)
        time_vector = time_vector.at[5 + current_step//self.time_discretization].set(1)
        time_vector = time_vector.at[current_match].set(1)

        ############# HANDLES mask_awake ##############
        mask_awake = obs.units.energy[team_id] > 0

        if team_id == 1:
            return {
                'image': jax.lax.stop_gradient(jax.vmap(mirror_grid)(image)),
                'vector': jax.lax.stop_gradient(rescaled_vector),
                'time': jax.lax.stop_gradient(time_vector),
                'position': jax.lax.stop_gradient(jax.vmap(mirror_position)(obs.units.position[team_id])),
                'mask_awake': jax.lax.stop_gradient(mask_awake),
                'action_mask': jax.lax.stop_gradient(get_action_masking_from_obs(team_id = team_id, obs = obs, sap_range=params.unit_sap_range)),
            }
        else:
            return {
                'image': jax.lax.stop_gradient(image),
                'vector': jax.lax.stop_gradient(rescaled_vector),
                'time': jax.lax.stop_gradient(time_vector),
                'position': jax.lax.stop_gradient(obs.units.position[team_id]),
                'mask_awake': jax.lax.stop_gradient(mask_awake),
                'action_mask': jax.lax.stop_gradient(get_action_masking_from_obs(team_id = team_id, obs = obs, sap_range=params.unit_sap_range)),
            }