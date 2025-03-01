from abc import ABC, abstractmethod
from luxai_s3.params import env_params_ranges
import gymnasium as gym
from typing import Any
import scipy
import numpy as np
from pytorch.env.utils import (
    mirror_grid, 
    mirror_position, 
    symmetrize, 
    manhattan_distance_to_nearest_point, 
    diagonal_distances, 
    Tiles, 
    get_action_masking_from_obs
)
from pytorch.env.utils import EnvParams
from agents.obs import Obs
from pytorch.env.memory import RelicPointMemoryState

class TransformObs(ABC):
    """
    Abstract base class for converting observations into tensor representations.
    """

    def __init__(self):
        pass

    def convert(
        self,
        team_id: int,
        obs: Obs,
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
        self.vector_std_values = np.array(list(self.vector_std.values()))
        self.vector_mean_values = np.array(list(self.vector_mean.values()))
        self.observation_space = gym.spaces.Dict({
                            'image': gym.spaces.Box(low=-1, high=1, shape=(self.image_channels, 24, 24), dtype=np.float32), 
                            'vector': gym.spaces.Box(low=-1, high=1, shape=(self.vector_size,), dtype=np.float32),
                            'time': gym.spaces.Box(low=0, high=1, shape=(self.time_size,), dtype=np.float32),
                            'position': gym.spaces.Box(low=0, high=23, shape=(16, 2), dtype=np.int8), 
                            'mask_awake': gym.spaces.Box(low=0, high=1, shape=(16,), dtype=np.int8),
                            #'action_mask': gym.spaces.Box(low=0, high=1, shape=(16, 6), dtype=np.int8),
        })

    def convert(self, team_id, obs, memory_state, params):
        image = np.zeros(
            (self.image_channels, 24, 24), 
            dtype=np.float32
        )

        vector = np.zeros((self.vector_size,), dtype=np.float32)

        ############# HANDLES IMAGE ##############
        image[0] = obs.sensor_mask
        image[1] = symmetrize(team_id, (obs.map_features.tile_type == Tiles.ASTEROID).astype(np.int8))
        image[2] = symmetrize(team_id, (obs.map_features.tile_type == Tiles.NEBULA).astype(np.int8))
        image[3] = symmetrize(team_id, (obs.map_features.energy / 20).astype(np.float32))

        # enemy units and ally units
        positions = obs.units.position
        image[4, positions[team_id, :, 0], positions[team_id, :, 1]] += obs.units_mask[team_id]
        image[5, positions[team_id, :, 0], positions[team_id, :, 1]] += np.maximum(1.0, obs.units.energy[team_id] + 1) / 400
        image[6, positions[1 - team_id, :, 0], positions[1 - team_id, :, 1]] += obs.units_mask[1 - team_id]
        image[7, positions[1 - team_id, :, 0], positions[1 - team_id, :, 1]] += np.maximum(1.0, obs.units.energy[1 - team_id] + 1) / 400

        image[8] = memory_state.relics_found_image
        # distance_matrix = symmetrize(team_id, manhattan_distance_to_nearest_point(memory_state.relics_found_positions, 24))
        # image[9] = distance_matrix / 24
        relic_circle = np.clip(scipy.signal.convolve2d(memory_state.relics_found_image == 1, np.ones((5, 5)), mode='same'), 0, 1)
        image[10] = relic_circle
        image[11] = memory_state.points_found_image
        image[12] = memory_state.last_visits_timestep / (obs.steps + 1)

        d1, d2, d3, d4, d5 = diagonal_distances(24)
        image[13] = d1 / 23
        image[14] = d2 / 23
        image[15] = d3 / 23
        image[16] = symmetrize(team_id, mirror_grid(d4)) / 23 if team_id == 1 else symmetrize(team_id, d4) / 23
        image[17] = symmetrize(team_id, mirror_grid(d5)) / 23 if team_id == 1 else symmetrize(team_id, d5) / 23

        ############# HANDLES VECTOR ##############
        vector[0] = params.unit_move_cost
        vector[1] = params.unit_sensor_range
        vector[2] = params.nebula_tile_vision_reduction
        vector[3] = params.nebula_tile_energy_reduction
        vector[4] = params.unit_sap_cost
        vector[5] = params.unit_sap_range
        vector[6] = params.unit_sap_dropoff_factor
        vector[7] = params.unit_energy_void_factor
        vector[8] = params.nebula_tile_drift_speed
        vector[9] = params.energy_node_drift_speed
        vector[10] = params.energy_node_drift_magnitude

        vector[11] = obs.team_points[team_id]
        vector[12] = obs.team_points[1 - team_id]
        vector[13] = memory_state.points_gained
        vector[14:17] = memory_state.relics_found_mask[0:3]

        rescaled_vector = (vector - self.vector_mean_values) / self.vector_std_values

        ############# HANDLES TIME WITH OHE ##############
        current_step = obs.steps % (params.max_steps_in_match + 1)
        current_match = obs.steps // (params.max_steps_in_match + 1)

        time_vector = np.zeros((55,), dtype=np.float32)
        time_vector[5 + current_step // self.time_discretization] = 1
        time_vector[current_match] = 1

        ############# HANDLES mask_awake ##############
        mask_awake = obs.units.energy[team_id] > 0

        if team_id == 1:
            return {
                'image': mirror_grid(image),
                'vector': rescaled_vector,
                'time': time_vector,
                'position': mirror_position(obs.units.position[team_id]),
                'mask_awake': mask_awake,
                #'action_mask': get_action_masking_from_obs(team_id, obs, params.unit_sap_range),
            }
        else:
            return {
                'image': image,
                'vector': rescaled_vector,
                'time': time_vector,
                'position': obs.units.position[team_id],
                'mask_awake': mask_awake,
                #'action_mask': get_action_masking_from_obs(team_id, obs, params.unit_sap_range),
            }
