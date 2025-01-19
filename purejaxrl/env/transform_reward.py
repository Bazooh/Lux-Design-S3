from abc import ABC, abstractmethod
from src.luxai_s3.state import EnvObs
import jax
from functools import partial
from luxai_s3.params import EnvParams, env_params_ranges
from purejaxrl.env.tracker import PlayerStats
from typing import Any
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
        player_stats: Any
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
        player_stats: PlayerStats
    ):
        return player_stats.points_gained

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
        player_stats: PlayerStats
    ):
        return player_stats.relics_discovered

class BasicFoundPointReward(TransformReward):
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
        player_stats: PlayerStats
    ):
        return player_stats.points_discovered 

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
        player_stats: dict
    ):
        return player_stats["cumulated_energy"]
