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
    @partial(jax.jit, static_argnums=(0,2))
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
        currently_viewing_relics = jnp.zeros((24,24), dtype = jnp.int8).at[obs.relic_nodes[:, 0], obs.relic_nodes[:, 1]].set(1)
        currently_viewing_relics = currently_viewing_relics.at[0, 0].set(0)
        currently_viewing_relics = currently_viewing_relics.at[23, 23].set(0)
        new_relics_found = new_relics_found + obs.sensor_mask * (
            (new_relics_found == 0) * (2 * currently_viewing_relics - 1)
        )
        points_gained = jnp.maximum(0.0, obs.team_points[team_id] - memory_state.last_step_team_points)
        new_points_awarding = memory_state.points_awarding
        
        ########## UPDATE RULE 2  ##########
        # Cases surrounded by no relics -> no points
        positions = obs.units.position[team_id]

        def process_out_of_range_unit(unit_pos):
            # Extract the neighborhood using dynamic_slice
            neighborhood = jax.lax.dynamic_slice(
                new_relics_found, 
                start_indices=(unit_pos[0]-2, unit_pos[1]-2), 
                slice_sizes=(5, 5)
            )
            no_relics = jnp.all(neighborhood == -1)
            is_alive = unit_pos.sum() > 0
            new_awarding_update = jnp.where(
                no_relics & is_alive, 
                -1, 
                new_points_awarding[unit_pos[0], unit_pos[1]]
            )
            return new_awarding_update
        updates = jax.vmap(process_out_of_range_unit)(positions)

        new_points_awarding = new_points_awarding.at[positions[:, 0], positions[:, 1]].set(updates)

        # RULE 3 AND 4 REFACTORED FOR JIT COMPATIBILITY
        active_points_awarding_mask = new_points_awarding[
            positions[:, 0], positions[:, 1]
        ]
        expected_gain = jnp.sum(active_points_awarding_mask == 1)
        unknown_points_mask_is_unknown = (active_points_awarding_mask == 0)

        new_points_awarding = jnp.where(
            points_gained == expected_gain, # Apply RULE 4:
            jnp.clip(new_points_awarding.at[positions[:, 0], positions[:, 1]].subtract(
                unknown_points_mask_is_unknown * obs.units_mask[team_id]),
                min = -1,
                max = 1
            ),            
            new_points_awarding
        )
        
        new_points_awarding = jnp.where(
            jnp.sum(unknown_points_mask_is_unknown) == (points_gained - expected_gain), # Apply RULE 3:
            jnp.clip(new_points_awarding.at[positions[:, 0], positions[:, 1]].add(
                unknown_points_mask_is_unknown * obs.units_mask[team_id]), 
                min = -1,
                max = 1
            ),
            new_points_awarding
        )

        return RelicPointMemoryState(
            relics_found=new_relics_found,
            points_awarding=new_points_awarding,
            last_step_team_points=obs.team_points[team_id],
            points_gained=points_gained,
        )
    
    def expand(self, obs, team_id, memory_state):
        return obs