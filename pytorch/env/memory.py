from abc import ABC, abstractmethod
from typing import cast
from dataclasses import dataclass
from luxai_s3.env import EnvObs, EnvParams
import numpy as np
import scipy
from typing import Any
from purejaxrl.env.utils import symmetrize, mirror_relic_positions_arrays

class Memory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, obs: EnvObs, team_id: int, memory_state: Any, params: EnvParams) -> Any: ...



    @abstractmethod
    def reset(self)-> Any: ...

class NoMemory(Memory):
    
    def update(self, obs: EnvObs, team_id: int, memory_state: Any, params: EnvParams) -> Any: 
        pass

    def reset(self):
        pass
   
@dataclass
class RelicPointMemoryState:
    relics_found_image: np.ndarray = np.zeros((24,24), dtype = np.int8) # -1 if there is no relic, 0 if unknown, 1 if there is a relic
    relics_found_mask: np.ndarray = np.zeros(6, dtype = np.bool_) # boolean array that indicates whether the relic has been found
    relics_found_positions: np.ndarray = -100 * np.ones((6,2), dtype = np.int32) # array of the positions of relic found so far, filled with -100
    last_visits_timestep: np.ndarray = np.zeros((24,24), dtype = np.int32) # the last time the square was viewed
    points_found_image: np.ndarray = np.zeros((24,24), dtype = np.int8) # -1 if the square can give a reward, 0 if unknown, 1 if the square gives a reward
    last_step_team_points: int = 0
    points_gained: int = 0

class RelicPointMemory:
    def reset(self):
        self.discovered_relics_id: set[int] = set()
        self.relic_nodes_mask = np.zeros(6, dtype=np.bool_)
        self.discovered_all_relics = False
        self.discovered_this_frame_id: set[int] = set()
        self.relic_tensor = np.zeros((24, 24), dtype=np.float32)
        self.relic_points = np.zeros((6, 2), dtype=np.int32)
        self.last_team_points: int = 0
        self.unknown_relics_tensor = np.zeros((24, 24), dtype=np.float32)
        self.unknown_points_tensor = np.zeros((24, 24), dtype=np.float32)
        self.discovered_all_points = False
        return RelicPointMemoryState(
            relics_found_image=self.unknown_relics_tensor,
            relics_found_mask=(self.relic_points).sum(axis=1) > 0,
            relics_found_positions=self.relic_points,
            points_found_image=self.unknown_points_tensor,
            last_step_team_points=0,
            points_gained=0,
            last_visits_timestep=self.unknown_points_tensor
        )
        
    def update(self, obs: EnvObs, team_id: int, params:EnvParams):
        if self.discovered_all_points:
            return
        self.unknown_relics_tensor += (
            (self.unknown_relics_tensor == 0)
            * obs.sensor_mask   
            * (2 * self.relic_tensor - 1)
        )

        team_points = cast(int, obs.team_points[team_id].item())
        points_gained = team_points - self.last_team_points
        self.last_team_points = team_points

        alive_units_id = obs.units_mask[team_id]
        if not alive_units_id.any():
            return

        # Cases surrounded by no relics -> no points
        unit_positions = np.array(obs.units.position[team_id])
        for unit_id in obs.get_available_units(team_id):
            unit_pos = unit_positions[unit_id]
            x, y = unit_pos[0].item(), unit_pos[1].item()

            if self.unknown_points_tensor[x, y] != 0:
                continue

            min_x = max(0, x - 2)
            max_x = min(24, x + 3)
            min_y = max(0, y - 2)
            max_y = min(24, y + 3)

            # If we are sure there are no relics around the unit, then this case earns no points
            if (self.unknown_relics_tensor[min_x:max_x, min_y:max_y] == -1).all():
                self.unknown_points_tensor[x, y] = -1

        alive_units_pos = obs.units.position[team_id][alive_units_id]
        unknown_points_mask = self.unknown_points_tensor[
            alive_units_pos[:, 0], alive_units_pos[:, 1]
        ]
        points_gained -= (unknown_points_mask == 1).sum()

        if points_gained == 0:
            self.unknown_points_tensor[
                alive_units_pos[:, 0], alive_units_pos[:, 1]
            ] -= (unknown_points_mask == 0).astype(np.int32)
        else:
            unknown_points_mask_is_unknown = (unknown_points_mask == 0).astype(np.int32)
            if unknown_points_mask_is_unknown.sum().item() == points_gained:
                self.unknown_points_tensor[
                    alive_units_pos[:, 0], alive_units_pos[:, 1]
                ] += unknown_points_mask_is_unknown

        if (self.unknown_points_tensor != 0).all():
            self.discovered_all_points = True
        
        return RelicPointMemoryState(
            relics_found_image=self.unknown_relics_tensor,
            relics_found_mask=(self.relic_points).sum(axis=1) > 0,
            relics_found_positions=self.relic_points,
            points_found_image=self.unknown_points_tensor,
            last_step_team_points=obs.team_points[team_id],
            points_gained=points_gained,
            last_visits_timestep=self.unknown_points_tensor
        )
