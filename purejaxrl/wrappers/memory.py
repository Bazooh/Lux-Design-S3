from abc import ABC, abstractmethod
from luxai_s3.env import EnvObs
import jax.numpy as jnp
from flax import struct
import jax, chex
from functools import partial
from typing import Any

class Memory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, obs: EnvObs, team_id: int, memory_state: Any) -> Any: ...

    @abstractmethod
    def expand(self, obs: EnvObs, team_id: int, memory_state: Any) -> EnvObs: ...

    @abstractmethod
    def reset(self)-> Any: ...

@struct.dataclass
class RelicPointMemoryState:
    relics_found: chex.Array
    """ -1 if there is no relic, 0 if unknown, 1 if there is a relic"""
    points_awarding: chex.Array
    """ -1 if the square can give a reward, 0 if unknown, 1 if the square gives a reward"""
    last_step_team_points: int

class RelicPointMemory(Memory):
    def __init__(self):
        pass

    def reset(self) -> RelicPointMemoryState:
        return RelicPointMemoryState(
                    relics_found=jnp.zeros((24,24), dtype = jnp.int8),
                    points_awarding= jnp.zeros((24,24), dtype = jnp.int8),
                    last_step_team_points=0
        )

    def update(self, obs: EnvObs, team_id: int, memory_state: RelicPointMemoryState):
        new_memory_state = self.reset()
        # new_memory_state.relics_found = memory_state.relics_found
        # new_memory_state.points_awarding = memory_state.points_awarding
        # new_memory_state.relics_found = memory_state.relics_found.at[3, obs.relic_nodes[:, 0], obs.relic_nodes[:, 1]].set(1)
        
        # alive_units_mask = obs.units_mask[team_id]
        # alive_units_pos = obs.units.position[team_id][alive_units_mask]
        # points_gained  = obs.team_points[team_id] - memory_state.last_step_team_points
        # new_memory_state.last_step_team_points = obs.team_points[team_id]
        return new_memory_state
    
    def expand(self, obs, team_id, memory_state):
        return obs