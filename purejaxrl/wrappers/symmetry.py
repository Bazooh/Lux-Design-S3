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
        team_id: int,
        obs: EnvObs,
    ):
        """
        Turn an assymetric obs to a symmetric one
        """
        pass

    @abstractmethod
    def convert_action(
        self,
        team_id: int,
        action
    ):
        """
        Turn an symetric action to a assymmetric one
        """
        pass    

@jax.vmap
def mirror_grid(array):
    """
    Input: (H, W)
    Output: (H, W)
    """
    return jnp.flip(jnp.transpose(array))

@jax.vmap
def mirror_position(pos):
    """
    Input: Shape (2): (x,y)
    Output: Shape 2: (23-y, 23-x)
    """
    return 23*jnp.ones(2) - jnp.flip(pos)

@jax.vmap
def mirror_action(a):
    # a is (3,)
    # 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left, 5 is sap
    flip_map = jnp.array([0, 3, 4, 1, 2]) 
    @jax.jit
    def flip_move_action(a):
        # a is (3,)
        # 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left
        a = a.at[0].set(flip_map[a[0]])  # Map the first element using flip_map
        return a

    @jax.jit
    def flip_sap_action(a):
        # a = (5, x, y), (x,y) should be replaced by (23-y, 13-x)
        a = a.at[1:].set(23*jnp.ones(2) - jnp.flip(a[1:]))
        return a
    
    a = jax.lax.cond(
        a[0] < 5,
        flip_move_action,
        flip_sap_action,
        a,
    )
    return a

class ActionAndObsSymmetry(Symmetry):
    def convert_obs(
        self,
        team_id: id,
        obs: EnvObs,
    ):
        if team_id == 0: # Do nothing
            return obs
        else:
            obs_attributes = list(obs.keys())
            if "image" in obs_attributes:
                obs["image"] = mirror_grid(obs["image"])
            if "position" in obs_attributes:
                obs["position"] = mirror_position(obs["position"]).astype(int)
            return obs


    def convert_action(
        self,
        team_id: id,
        action
    ):
        if team_id == 0: # Do nothing
            return action
        else:
            return mirror_action(action).astype(int)

