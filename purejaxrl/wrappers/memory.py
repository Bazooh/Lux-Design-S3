from abc import ABC, abstractmethod
from luxai_s3.env import EnvObs
import jax.numpy as jnp
from flax import struct
import jax, chex
from functools import partial
from typing import Any
import numpy as np
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
    points_gained: int

class RelicPointMemory(Memory):
    def __init__(self):
        pass

    def reset(self) -> RelicPointMemoryState:
        return RelicPointMemoryState(
                    relics_found=jnp.zeros((24,24), dtype = jnp.int8),
                    points_awarding= jnp.zeros((24,24), dtype = jnp.int8),
                    last_step_team_points=0,
                    points_gained=0,

        )
    #@partial(jax.jit, static_argnums=(0,2))
    def update(self, obs: EnvObs, team_id: int, memory_state: RelicPointMemoryState):
        """
        Update rule 1:
        - Update the relics_found to 1 if a relic is discovered.
        Update rule 2:
        - If a unit is oob of relic, then set the points_awarding of the units position to -1.
        Update rule 3:
        - If N is the number of the units in range of a relic, and we have gained N points, then set the points_awarding of all the units positions to 1.
        Update rule 4:
        - If I sit on exactly N points square (N farming units), and I win exactly N points, then set the non farming ones to -1
        """    

        ########## UPDATE RULE 1  ##########
        new_relics_found = memory_state.relics_found
        viewed_relic_obs_indices = obs.get_avaible_relics()
        currently_viewing_relics = jnp.zeros((24,24), dtype = jnp.int8).at[obs.relic_nodes[viewed_relic_obs_indices, 0], obs.relic_nodes[viewed_relic_obs_indices, 1]].set(1)
        new_relics_found = new_relics_found + obs.sensor_mask * (
            (new_relics_found == 0) * (2 * currently_viewing_relics - 1)
        )
        points_gained = max(0, obs.team_points[team_id] - memory_state.last_step_team_points)

        ########## UPDATE RULE 2  ##########
        # Cases surrounded by no relics -> no points
        new_points_awarding = memory_state.points_awarding
        alive_units_id = obs.get_avaible_units(team_id)
        unit_positions = np.array(obs.units.position[team_id])
        for unit_id in alive_units_id:
            unit_pos = unit_positions[unit_id]
            x, y = unit_pos[0].item(), unit_pos[1].item()

            if memory_state.points_awarding[x, y] != 0:
                continue

            min_x = max(0, x - 2)
            max_x = min(24, x + 3)
            min_y = max(0, y - 2)
            max_y = min(24, y + 3)

            # If we are sure there are no relics around the unit, then this case earns no points !
            if (new_relics_found[min_x:max_x, min_y:max_y] == -1).all():
                new_points_awarding = new_points_awarding.at[x,y].set(-1)


        ########## UPDATE RULE 3 AND 4  ##########
        alive_units_pos = obs.units.position[team_id][alive_units_id]
        unknown_points_mask = new_points_awarding[
            alive_units_pos[:, 0], alive_units_pos[:, 1]
        ]
        expected_gain = (unknown_points_mask == 1).sum().item()
        unknown_points_mask_is_unknown = (unknown_points_mask == 0)
        
        if points_gained == expected_gain: # RULE 4
            new_points_awarding = new_points_awarding.at[
                alive_units_pos[:, 0], alive_units_pos[:, 1]
            ].subtract(unknown_points_mask_is_unknown)
        else:
            if unknown_points_mask_is_unknown.sum().item() == points_gained - expected_gain: # RULE 3
               new_points_awarding = new_points_awarding.at[
                    alive_units_pos[:, 0], alive_units_pos[:, 1]
                ].add(unknown_points_mask_is_unknown)

        return RelicPointMemoryState(
            relics_found=new_relics_found,
            points_awarding=new_points_awarding,
            last_step_team_points=obs.team_points[team_id],
            points_gained=points_gained,
        )
    
    def expand(self, obs, team_id, memory_state):
        return obs