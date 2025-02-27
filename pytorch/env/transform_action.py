from abc import ABC, abstractmethod
import gymnasium as gym
from luxai_s3.env import EnvObs, EnvParams
from purejaxrl.env.utils import mirror_action, get_full_sap_action
from typing import Any
import numpy as np
PlayerAction = Any
from agents.obs import Obs
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
        
class SimplerActionWithSap(TransformAction):
    def __init__(self, do_mirror_input = True):
        super().__init__()
        self.action_space = gym.spaces.Discrete(6)
        self.do_mirror_input = do_mirror_input # whether to mirror the input (from [0, 1, 2, 3, 4, 5] to [0, 3, 1, 4, 2, 5])
    
    def convert(
        self,
        team_id: int,
        action: PlayerAction, #shape = 16
        obs: Obs,
        params: EnvParams,
    ):
        if self.do_mirror_input:
            action = np.array([mirror_action(action[i]) for i in range(16)]) if team_id == 1 else action
        
        sap_deltas = np.zeros((16,2), dtype=np.int32)
        sap_deltas[:,0] = -1 #add 1 to the x position

        new_action = np.concatenate([action.reshape(16,1), sap_deltas], axis=1)

        sap_range = params.unit_sap_range#int
        enemy_team_id = 1 - team_id
        enemy_positions = obs.units.position[enemy_team_id, :, :] #16,2
        ally_positions = obs.units.position[team_id, :, :] #16,2
        ally_actions = new_action[:,0]

        new_action = np.array([
            get_full_sap_action(ally_actions[i], ally_positions[i,0],ally_positions[i,1],enemy_positions,sap_range)
            for i in range(16)
        ])
        
        return new_action
        
    
    