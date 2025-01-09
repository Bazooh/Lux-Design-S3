from abc import ABC, abstractmethod
from src.luxai_s3.state import EnvObs
import jax
from functools import partial
from luxai_s3.params import EnvParams, env_params_ranges

class TransformReward(ABC):
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
        team_id_str: str,
        obs: EnvObs,
        params: EnvParams,
        reward: float,
    ):
        #return reward
        return reward