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
        team_id_str: str,
        last_obs: EnvObs,
        obs: EnvObs,
        params: EnvParams,
        reward: float,
    ):
        """
        Converts a reward into a reshaped reward
        """
        pass
        
class BasicPointBasedReward(TransformReward):
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
        return obs.team_points[team_id] - last_obs.team_points[team_id]