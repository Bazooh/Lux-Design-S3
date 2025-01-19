from abc import ABC, abstractmethod
from src.luxai_s3.state import EnvObs
import jax
from functools import partial
from luxai_s3.params import EnvParams, env_params_ranges

class TransformReward(ABC):
    """
    Abstract base class for reshaping rewards
    """

    def __init__(self):
        pass


    @abstractmethod
    def convert(
        self,
        team_id: int,
        last_obs: EnvObs,
        obs: EnvObs,
        params: EnvParams,
        reward: float,
    ):
        """
        Converts a reward into a reshaped reward
        """
        pass
        
class BasicPointReward(TransformReward):
    def __init__(self):
        super().__init__()
        pass
    @partial(jax.jit, static_argnums=(0, 1))
    def convert(
        self,
        team_id: int,
        obs: EnvObs,
        last_obs: EnvObs,
        params: EnvParams,
        reward: float,
    ):
        points_gained = jax.numpy.maximum(0, obs.team_points[team_id] - last_obs.team_points[team_id])
        units_mask = jax.numpy.sum(obs.units_mask[team_id])
        return points_gained

class BasicExplorationReward(TransformReward):
    def __init__(self):
        super().__init__()
        pass
    @partial(jax.jit, static_argnums=(0, 1))
    def convert(
        self,
        team_id: int,
        obs: EnvObs,
        last_obs: EnvObs,
        params: EnvParams,
        reward: float,
    ):
        return jax.numpy.sum(jax.numpy.clip(obs.sensor_mask.astype(jax.numpy.int8) - last_obs.sensor_mask.astype(jax.numpy.int8), min=0, max=1))

class BasicFoundRelicReward(TransformReward):
    def __init__(self):
        super().__init__()
        pass
    @partial(jax.jit, static_argnums=(0, 1))
    def convert(
        self,
        team_id: int,
        obs: EnvObs,
        last_obs: EnvObs,
        params: EnvParams,
        reward: float,
    ):
        return jax.numpy.sum(jax.numpy.clip(obs.relic_nodes_mask.astype(jax.numpy.int8) - last_obs.relic_nodes_mask.astype(jax.numpy.int8), min=0, max=1))
      

class BasicEnergyReward(TransformReward):
    def __init__(self):
        super().__init__()
        pass
    @partial(jax.jit, static_argnums=(0, 1))
    def convert(
        self,
        team_id: int,
        obs: EnvObs,
        last_obs: EnvObs,
        params: EnvParams,
        reward: float,
    ):
        energy_map = jax.numpy.zeros((obs.map_features.energy.shape[0], obs.map_features.energy.shape[1]), dtype=jax.numpy.float32)
        energy_map = energy_map.at[
            obs.units.position[1- team_id, :, 0],
            obs.units.position[1- team_id, :, 1],
        ].set(obs.units.energy[1 -team_id] + 1) / 400

        return jax.numpy.sum(energy_map)
