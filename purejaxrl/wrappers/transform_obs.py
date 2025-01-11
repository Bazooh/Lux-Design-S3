from abc import ABC, abstractmethod

import gymnax.environments.spaces
from agents.lux.utils import Tiles
from src.luxai_s3.state import EnvObs
import jax
from functools import partial
from luxai_s3.params import EnvParams, env_params_ranges
import gymnax

class TransformObs(ABC):
    """
    Abstract base class for converting observations into tensor representations.
    """

    def __init__(self, symmetry: bool = True):
        self.symmetry = symmetry
        pass


    @abstractmethod
    def convert(
        self,
        team_id_str: str,
        obs: EnvObs,
        params: EnvParams,
        reward: float,
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

    def __init__(self, symmetry: bool = True):
        super().__init__(symmetry)
        self.image_features = { # Key: Name of the feature, Value: Number of channels required representing the feature
            "Unknown": 1,
            "Asteroid": 1,
            "Nebula": 1,
            "Relic": 1,
            "Energy_Field": 1,
            "Enemy_Units": 1,
            "Ally_Units": 1,
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
            # Lead in points
            "current_points": 1, # 1 float
            # Current position (x,y)
            "current_position": 2, # 2 floats
            # Relic Positions
            "relic_relative_positions": 12, # maximum 6 relic nodes, 2 floats each
            # Unit Energy
            "unit_energy": 1, # 1 float for your energy
        }
        self.vector_size = sum(self.vector_features.values())
        self.image_channels = sum(self.vector_features.values())
        self.observation_space = gymnax.environments.spaces.Dict({
                            'image': gymnax.environments.spaces.Box(low=0, high=1, shape=(self.image_channels, 24, 24), dtype=jax.numpy.float32), 
                            'vector': gymnax.environments.spaces.Box(low=-1, high=1, shape=(self.vector_size), dtype=jax.numpy.float32),
                            'position': gymnax.environments.spaces.Box(low=0, high=23, shape=(16, 2), dtype=jax.numpy.int8), 
        })
    
    @partial(jax.jit, static_argnums=(0, 1))
    def convert(
        self, 
        team_id_str: str,
        obs: EnvObs, 
        reward: float,
        params: EnvParams,
    ):
        team_id = int(team_id_str[-1])
        image = jax.numpy.zeros(
            (self.image_channels, obs.map_features.energy.shape[0], obs.map_features.energy.shape[1]),
            dtype=jax.numpy.float32,
        ) 

        vector = jax.numpy.zeros(
            (self.vector_size), 
            dtype=jax.numpy.float32
        )
    
        ############# HANDLES IMAGE ##############
        image = image.at[0].set(obs.sensor_mask) # unknown
        image = image.at[1].set(obs.map_features.tile_type == Tiles.ASTEROID) # asteroids
        image = image.at[2].set(obs.map_features.tile_type == Tiles.NEBULA) # nebula
        image = image.at[3, obs.relic_nodes[:, 0], obs.relic_nodes[:, 1]].set(1)# relics
        image = image.at[4].set(obs.map_features.energy / 20) # energy field

        # enemy units and ally units
        positions = jax.numpy.array(obs.units.position)
        image = image.at[
            5,
            positions[1 - team_id, :, 0],
            positions[1 - team_id, :, 1],
        ].set((obs.units.energy[1 - team_id] + 1) / 400)

        image = image.at[
            6,
            positions[team_id, :, 0],
            positions[team_id, :, 1],
        ].set(obs.units.energy[team_id] + 1) / 400


        ############# GET INDIVIDUAL VECTORS ##############

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
        return {'image': image, 'vector': vector, 'position': obs.units.position[team_id]}