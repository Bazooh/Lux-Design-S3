from abc import ABC, abstractmethod
from src.luxai_s3.state import EnvObs
import jax
from functools import partial
from luxai_s3.params import EnvParams, env_params_ranges
import jax.numpy as jnp

class Symmetry(ABC):
    """
    Abstract base class for symmetrizing actions and observations
    """

    def __init__(self):
        pass


    @abstractmethod
    def convert_obs(
        self,
        team_id_str: str,
        obs: EnvObs,
    ):
        """
        Turn an assymetric obs to a symmetric one
        """
        pass

    @abstractmethod
    def convert_action(
        self,
        team_id_str: str,
        action
    ):
        """
        Turn an symetric action to a assymmetric one
        """
        pass    

@jax.jit
def mirror_grid(array):
    """
    Input: (N, H, W)
    Output: (N, H, W)
    """
    return jnp.flip(jnp.transpose(array, axes=(0, 2, 1)), axis=(1, 2))

@jax.jit
def mirror_action(a):
    # a is (3,)
    # 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left, 5 is sap

    @jax.jit
    def flip_move_action(a):
        # a is (3,)
        # 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left
        a = a.at[0].set([0, 3, 4, 1, 2][a[0]])
        return a

    @jax.jit
    def flip_sap_action(a):
        # a = (5, x, y), (x,y) should be replaced by (-y, -x)
        a = a.at[1:2].set(-a[2], -a[1])
        return a
    
    a = jax.lax.cond(
        a < 5,
        lambda: flip_move_action,
        lambda: flip_sap_action,
        a,
    )
    return a

class ActionAndObsSymmetry(Symmetry):
    def convert_obs(
        self,
        team_id_str: str,
        obs: EnvObs,
    ):
        return obs


    def convert_action(
        self,
        team_id_str: str,
        action
    ):
        return action

