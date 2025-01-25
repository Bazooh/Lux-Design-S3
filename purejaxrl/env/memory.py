from abc import ABC, abstractmethod
from luxai_s3.env import EnvObs
import jax.numpy as jnp
from flax import struct
import jax, chex
from functools import partial
from typing import Any
from purejaxrl.utils import symmetrize
class Memory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, obs: EnvObs, team_id: int, memory_state: Any) -> Any: ...



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
        - If I sit on exactly N points square (N farming units), and I win exactly N + M points, and M is the number of units that could be farming, 
        then set the non farming ones to -1
        """    

        ########## UPDATE RULE 1  ##########
        new_relics_found = memory_state.relics_found
        currently_viewing_relics = jnp.zeros((24,24), dtype = jnp.int8).at[obs.relic_nodes[:, 0], obs.relic_nodes[:, 1]].set(1)
        currently_viewing_relics = currently_viewing_relics.at[0, 0].set(0)
        currently_viewing_relics = currently_viewing_relics.at[23, 23].set(0)
        new_relics_found = new_relics_found + obs.sensor_mask * (
            (new_relics_found == 0) * (2 * currently_viewing_relics - 1)
        ) +  obs.sensor_mask * 2 * (new_relics_found == -1) * currently_viewing_relics
        points_gained = jnp.maximum(0, obs.team_points[team_id] - memory_state.last_step_team_points)
        new_points_awarding = memory_state.points_awarding
        
        ########## UPDATE RULE 2  ##########
        positions = obs.units.position[team_id]

        def is_out_of_range_unit(idx):
            # Extract the neighborhood using dynamic_slice
            unit_pos = positions[idx]
            neighborhood = jax.lax.dynamic_slice(
                new_relics_found, 
                start_indices=(unit_pos[0]-2, unit_pos[0]-2), 
                slice_sizes=(5, 5)
            )
            no_relics_around = jnp.all(neighborhood == -1)
            return no_relics_around
        
        units_out_of_range = jax.vmap(is_out_of_range_unit)(jnp.arange(16))
        alive_units_image = jnp.zeros((24,24), dtype = jnp.bool).at[positions[:, 0], positions[:, 1]].set(obs.units.energy[team_id] > 0)
        units_out_of_range_image = jnp.zeros((24,24), dtype = jnp.bool).at[positions[:, 0], positions[:, 1]].set(units_out_of_range) & alive_units_image
        farming_units_image = (new_points_awarding == 1) & alive_units_image
        
        new_points_awarding = jnp.where(units_out_of_range_image, -1, new_points_awarding)

        ########## UPDATE RULE 3 ##########
        # If I sit on exactly N points square (N farming units), and I win exactly N + M points, 
        # and M is the number of units that could be farming, then set the non farming ones to -1
        unknown_farming_units_image = (new_points_awarding == 0) & alive_units_image
        M = jnp.sum(unknown_farming_units_image.astype(jnp.int8))
        N = jnp.sum(farming_units_image.astype(jnp.int8))
        new_points_awarding = jax.lax.cond(
            points_gained == N + M,
            lambda:  jnp.where(unknown_farming_units_image, -1, new_points_awarding),
            lambda: new_points_awarding,
        )

        ########## SYMMETRIZE  ##########
        new_relics_found =  symmetrize(team_id, new_relics_found)
        new_points_awarding = symmetrize(team_id, new_points_awarding)

        return RelicPointMemoryState(
            relics_found=jax.lax.stop_gradient(new_relics_found),
            points_awarding=jax.lax.stop_gradient(new_points_awarding),
            last_step_team_points=obs.team_points[team_id],
            points_gained=points_gained,
        )
