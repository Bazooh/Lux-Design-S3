from abc import ABC, abstractmethod

import gymnax.environments.spaces
from luxai_s3.env import EnvObs, EnvParams, PlayerAction
import jax
import jax.numpy as jnp
from functools import partial
import gymnax
from purejaxrl.env.utils import mirror_action, get_full_sap_action
class TransformAction(ABC):
    """
    Abstract base class for converting simple actions (ie flat 16) to complete actions (16,3).
    """

    def __init__(self):
        pass


    @abstractmethod
    def convert(
        self,
        team_id: int,
        action: PlayerAction,
        obs: EnvObs,
        params: EnvParams,
    ):
        """
        Converts an action 
        """
        pass
        
class SimplerActionNoSap(TransformAction):
    def __init__(self, do_mirror = True):
        super().__init__()
        self.action_space = gymnax.environments.spaces.Discrete(6)
        self.do_mirror = do_mirror
    
    @partial(jax.jit, static_argnums=(0, 1))
    def convert(
        self,
        team_id: int,
        action: PlayerAction,
        obs: EnvObs,
        params: EnvParams,
    ):
        new_action = jax.numpy.zeros((16,3), dtype=jax.numpy.int32)
        new_action = new_action.at[:,0].set(action)

        if self.do_mirror:
            return jax.vmap(mirror_action)(new_action) if team_id == 1 else new_action
        else :
            return new_action
    


class SimplerActionWithSap(TransformAction):
    def __init__(self, do_mirror = True):
        super().__init__()
        self.action_space = gymnax.environments.spaces.Discrete(6)
        self.do_mirror = do_mirror
    
    @partial(jax.jit, static_argnums=(0, 1))
    def convert(
        self,
        team_id: int,
        action: PlayerAction, #shape = 16
        obs: EnvObs,
        params: EnvParams,
    ):
        sap_deltas = jnp.zeros((16,2), dtype=jnp.int32)
        sap_deltas = sap_deltas.at[:,0].set(-1) #add 1 to the x position

        new_action = jnp.concatenate([action.reshape(16,1), sap_deltas], axis=1)

        sap_range = params.unit_sap_range#int
        enemy_team_id = 1 - team_id
        enemy_positions = obs.units.position[enemy_team_id, :, :] #16,2
        ally_positions = obs.units.position[team_id, :, :] #16,2
        ally_actions = new_action[:,0]

        new_action = jax.vmap(get_full_sap_action, in_axes = (0,0,0,None,None))(ally_actions, ally_positions[:,0],ally_positions[:,1],enemy_positions,sap_range)
        jax.debug.print("new_actions : {p}", p = new_action)


        if self.do_mirror:
            return jax.vmap(mirror_action)(new_action) if team_id == 1 else new_action
        else :
            return new_action

        
    
    