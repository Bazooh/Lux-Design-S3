from abc import ABC, abstractmethod

import gymnax.environments.spaces
from luxai_s3.env import EnvObs, EnvParams, PlayerAction
import jax
from functools import partial
import gymnax
class TransformAction(ABC):
    """
    Abstract base class for converting simple actions (ie flat 16) to complete actions (16,3).
    """

    def __init__(self):
        pass


    @abstractmethod
    def convert(
        self,
        team_id_str: str,
        action: PlayerAction,
        obs: EnvObs,
        params: EnvParams,
    ):
        """
        Converts an action 
        """
        pass
        
class SimplerActionNoSap(TransformAction):
    def __init__(self):
        super().__init__()
        self.action_space = gymnax.environments.spaces.Discrete(6)
    @partial(jax.jit, static_argnums=(0, 1))
    def convert(
        self,
        team_id_str: str,
        action: PlayerAction,
        obs: EnvObs,
        params: EnvParams,
    ):
        new_action = jax.numpy.zeros((16,3), dtype=jax.numpy.int16)
        new_action = new_action.at[:,0].set(action)
        return new_action
        

    