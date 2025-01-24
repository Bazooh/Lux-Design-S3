from abc import ABC, abstractmethod

import gymnax.environments.spaces
from agents.lux.utils import Tiles
from src.luxai_s3.state import EnvObs
import jax
from functools import partial
from luxai_s3.params import EnvParams, env_params_ranges
import gymnax
from typing import Any
import jax.numpy as jnp
import numpy as np
import chex
from purejaxrl.utils import mirror_grid, mirror_position

class TransformObs(ABC):
    """
    Abstract base class for converting observations into tensor representations.
    """

    def __init__(self):
        pass


    @abstractmethod
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
        self.image_features = { # Key: Name of the feature, Value: Number of channels required representing the feature
            "Unknown": 1,
            "Asteroid": 1,
            "Nebula": 1,
            "Energy_Field": 1,
            "Ally_Units": 1,
            "Enemy_Units": 1,
            "Relic": 1,
            "Points": 1,
        }
        self.vector_features = { # Key: Name of the feature, Value: Size of the vector representing the feature
            # Game Parameters
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
            "team_points": 1, # 1 float
            "opponent_points": 1, # 1 float
            "points_gained": 1, # 1 float
            "steps": 1, # 1 float
        }
        self.vector_size = sum(self.vector_features.values())
        self.image_channels = sum(self.image_features.values())

        self.vector_mean = {
            "unit_move_cost": np.mean(env_params_ranges["unit_move_cost"]), 
            "unit_sensor_range": np.mean(env_params_ranges["unit_sensor_range"]), 
            "nebula_tile_vision_reduction": np.mean(env_params_ranges["nebula_tile_vision_reduction"]),
            "nebula_tile_energy_reduction": np.mean(env_params_ranges["nebula_tile_energy_reduction"]),
            "unit_sap_cost": np.mean(env_params_ranges["unit_sap_cost"]),
            "unit_sap_range": np.mean(env_params_ranges["unit_sap_range"]),
            "unit_sap_dropoff_factor": np.mean(env_params_ranges["unit_sap_dropoff_factor"]),
            "unit_energy_void_factor": np.mean(env_params_ranges["unit_energy_void_factor"]),
            "nebula_tile_drift_speed": np.mean(env_params_ranges["nebula_tile_drift_speed"]),
            "energy_node_drift_speed": np.mean(env_params_ranges["energy_node_drift_speed"]),
            "energy_node_drift_magnitude": np.mean(env_params_ranges["energy_node_drift_magnitude"]),
            "team_points": 0,
            "opponent_points": 0,
            "points_gained": 0,
            "steps": 0
        }

        self.vector_std = {
            "unit_move_cost": np.std(env_params_ranges["unit_move_cost"]), 
            "unit_sensor_range": np.std(env_params_ranges["unit_sensor_range"]), 
            "nebula_tile_vision_reduction": np.std(env_params_ranges["nebula_tile_vision_reduction"]),
            "nebula_tile_energy_reduction": np.std(env_params_ranges["nebula_tile_energy_reduction"]),
            "unit_sap_cost": np.std(env_params_ranges["unit_sap_cost"]),
            "unit_sap_range": np.std(env_params_ranges["unit_sap_range"]),
            "unit_sap_dropoff_factor": np.std(env_params_ranges["unit_sap_dropoff_factor"]),
            "unit_energy_void_factor": np.std(env_params_ranges["unit_energy_void_factor"]),
            "nebula_tile_drift_speed": np.std(env_params_ranges["nebula_tile_drift_speed"]),
            "energy_node_drift_speed": np.std(env_params_ranges["energy_node_drift_speed"]),
            "energy_node_drift_magnitude": np.std(env_params_ranges["energy_node_drift_magnitude"]),
            "team_points": 1,
            "opponent_points": 1,
            "points_gained": 1,
            "steps": 1
        }

        self.vector_std_values = jnp.array(list(self.vector_std.values()))
        self.vector_mean_values = jnp.array(list(self.vector_mean.values()))
        self.observation_space = gymnax.environments.spaces.Dict({
                            'image': gymnax.environments.spaces.Box(low=-1, high=1, shape=(self.image_channels, 24, 24), dtype=jnp.int8), 
                            'vector': gymnax.environments.spaces.Box(low=-1, high=1, shape=(self.vector_size), dtype=jnp.float32),
                            'position': gymnax.environments.spaces.Box(low=0, high=23, shape=(16, 2), dtype=jnp.int8), 
        })
    @partial(jax.jit, static_argnums=(0, 1))
    def convert(
        self, 
        team_id: int,
        obs: EnvObs, 
        memory_state: Any,
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
        mask_unseen = jnp.logical_not(obs.sensor_mask)
        image = image.at[0].set(obs.sensor_mask) # unknown
        image = image.at[1].set(obs.map_features.tile_type == Tiles.ASTEROID) # asteroids
        image = image.at[2].set(obs.map_features.tile_type == Tiles.NEBULA) # nebula
        image = image.at[3].set(obs.map_features.energy / 20) # energy field

        image = image.at[1, :].add(jnp.multiply(jnp.flip(image[1], axis=(0, 1)).T,mask_unseen)) # symmetrize asteroid
        image = image.at[2, :].add(jnp.multiply(jnp.flip(image[2], axis=(0, 1)).T,mask_unseen)) # symmetrize nebula
        image = image.at[3, :].add(jnp.multiply(jnp.flip(image[3], axis=(0, 1)).T,mask_unseen)) # symmetrize energy field

        # enemy units and ally units
        positions = jnp.array(obs.units.position)
        image = image.at[
            4,
            positions[team_id, :, 0],
            positions[team_id, :, 1],
        ].set((obs.units.energy[team_id] + 1) / 400)

        image = image.at[
            5,
            positions[1- team_id, :, 0],
            positions[1- team_id, :, 1],
        ].set(obs.units.energy[1 -team_id] + 1) / 400

        image = image.at[6].set(memory_state.relics_found)
        image = image.at[7].set(memory_state.points_awarding)    
        image = image.at[6, :].add(jnp.flip(image[6] == 1, axis=(0, 1)).T) # symmetrize relics
        image = image.at[7, :].add(jnp.flip(image[7] == 1, axis=(0, 1)).T)
        image = image.at[6, :].set(jnp.clip(image[6], -1, 1))
        image = image.at[7, :].set(jnp.clip(image[7], -1, 1))

        # Game parameters
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

        # current_points
        vector= vector.at[11].set(obs.team_points[team_id])
        vector= vector.at[12].set(obs.team_points[1-team_id])
        vector= vector.at[13].set(memory_state.points_gained)
        vector= vector.at[14].set(obs.steps)
        rescaled_vector = (vector - self.vector_mean_values) / self.vector_std_values
        

        if team_id == 1:
            return {
                'image': jax.lax.stop_gradient(jax.vmap(mirror_grid)(image)),
                'vector': jax.lax.stop_gradient(rescaled_vector),
                'position': jax.lax.stop_gradient(jax.vmap(mirror_position)(obs.units.position[team_id])),
            }
        else:
            return {
                'image': jax.lax.stop_gradient(image),
                'vector': jax.lax.stop_gradient(rescaled_vector),
                'position': jax.lax.stop_gradient(obs.units.position[team_id]),
            }