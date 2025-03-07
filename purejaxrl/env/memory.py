from abc import ABC, abstractmethod

from luxai_s3.env import EnvObs, EnvParams
import jax.numpy as jnp
from flax import struct
import jax, chex
from functools import partial
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
   
 
@struct.dataclass
class RelicPointMemoryState:
    relics_found_image: chex.Array = jnp.zeros((24,24), dtype = jnp.int8) # -1 if there is no relic, 0 if unknown, 1 if there is a relic
    relics_found_mask: chex.Array = jnp.zeros(6, dtype = jnp.bool_) # boolean array that indicates whether the relic has been found
    relics_found_positions: chex.Array = -100 * jnp.ones((6,2), dtype = jnp.int32) # array of the positions of relic found so far, filled with -100
    last_visits_timestep: chex.Array = jnp.zeros((24,24), dtype = jnp.int32) # the last time the square was viewed
    points_found_image: chex.Array = jnp.zeros((24,24), dtype = jnp.int8) # -1 if the square can give a reward, 0 if unknown, 1 if the square gives a reward
    last_step_team_points: int = 0
    points_gained: int = 0

class RelicPointMemory(Memory):
    def __init__(self):
        pass

    def reset(self) -> RelicPointMemoryState:
        return RelicPointMemoryState()
    @partial(jax.jit, static_argnums=(0,2))
    def update(self, obs: EnvObs, team_id: int, memory_state: RelicPointMemoryState, params: EnvParams):
        """
        Update rule 1:
        - Update the relics_found_image to 1 if a relic is discovered, and the points_found_image around to max(0, points_found_image)
        Update rule 2:
        - If a unit is oob of relic, then set the points_found_image of the units position to -1.
        Update rule 3:
        - If I sit on exactly N points square (N farming units), and I win exactly N + M points, and M is the number of units that could be farming, 
        then set the units that could be farming to points_found_image = 1 (only if obs.steps != (max_steps_in_match + 1) )
        Update rule 4:
        - If I sit on exactly N points square (N farming units), and I win exactly N points,
        # then set the units the other units to points_found_image = -1  (only if obs.steps !=  (max_steps_in_match + 1) )
        """
        new_last_visits_timestep = memory_state.last_visits_timestep + jnp.ones((24,24), dtype = jnp.int32)
        new_last_visits_timestep = jnp.where(obs.sensor_mask == 1, 0, new_last_visits_timestep)
        new_relics_found_image = memory_state.relics_found_image
        new_points_found_image = memory_state.points_found_image
        points_gained = jnp.maximum(0, obs.team_points[team_id] - memory_state.last_step_team_points)

        ########## UPDATE RULE 1  ##########
        currently_viewing_relics_image = jnp.zeros((24,24), dtype = jnp.int8).at[obs.relic_nodes[:, 0], obs.relic_nodes[:, 1]].set(1)
        currently_viewing_relics_image = currently_viewing_relics_image.at[0, 0].set(0)
        currently_viewing_relics_image = currently_viewing_relics_image.at[23, 23].set(0)
        currently_viewing_relics_image = symmetrize(team_id, currently_viewing_relics_image)
        
        discovered_relics_image = (currently_viewing_relics_image == 1) & (new_relics_found_image <= 0)
        cells_around_discovery = jax.scipy.signal.convolve2d(discovered_relics_image,jnp.ones((5, 5)), mode='same')
        cells_around_discovery = cells_around_discovery>0
        cells_around_discovery = symmetrize(team_id, cells_around_discovery.astype(jnp.int8)).astype(jnp.bool_)

        new_relics_found_image = jnp.where(discovered_relics_image, 1, new_relics_found_image) # I set the cells where i discovered a relic to 1
        new_relics_found_image = jnp.where(obs.sensor_mask & ~(new_relics_found_image == 1), -1, new_relics_found_image) # I set the cells that i see and dont have relics to -1
        new_relics_found_image = symmetrize(team_id, new_relics_found_image)
        
        new_relics_found_positions = jnp.where(obs.relic_nodes > 0, obs.relic_nodes, memory_state.relics_found_positions)
        new_relics_found_positions = mirror_relic_positions_arrays(new_relics_found_positions)
        new_relics_found_mask = jnp.sum(new_relics_found_positions, axis = -1) > 0

        new_points_found_image = jnp.where(cells_around_discovery&~(new_points_found_image == 1), 0, new_points_found_image) # I set the cells near a discovered relic to points_found_image = 0
        
        ########## UPDATE RULE 2  ##########
        positions = obs.units.position[team_id]
        alive_units_image = jnp.zeros((24,24), dtype = jnp.int8).at[positions[:, 0], positions[:, 1]].set(1)
        alive_units_image = (alive_units_image == 1)
        could_be_relic_image = (new_relics_found_image >= 0).astype(jnp.int8)
        cells_in_relic_range_image = jax.scipy.signal.convolve2d(could_be_relic_image, jnp.ones((5, 5)), mode='same')
        cells_in_relic_range_image = cells_in_relic_range_image>0
        units_out_of_range_image = (~cells_in_relic_range_image) & alive_units_image

        new_points_found_image = jnp.where(units_out_of_range_image, -1, new_points_found_image) # I set the units that are out of range to points_found_image = -1

        ########## UPDATE RULE 3 ##########
        # If I sit on exactly N points square (N farming units), and I win exactly N + M points, and M is the number of units that could be farming, 
        # then set the units that could be farming to points_found_image = 1
        farming_units_image = (new_points_found_image == 1) & alive_units_image
        unknown_farming_units_image = (new_points_found_image == 0) & cells_in_relic_range_image & alive_units_image
        M = jnp.sum(unknown_farming_units_image.astype(jnp.int8))
        N = jnp.sum(farming_units_image.astype(jnp.int8))
        new_points_found_image = jax.lax.cond(
            (points_gained == (N + M)) & (obs.steps % (params.max_steps_in_match + 1) != 0),
            lambda:  jnp.where(unknown_farming_units_image > 0, 1, new_points_found_image),
            lambda: new_points_found_image,
        )

        ########## UPDATE RULE 4 ##########
        # If I sit on exactly N points square (N farming units), and I win exactly N points,
        # then set the units the other units to points_found_image = -1
        farming_units_image = (new_points_found_image == 1) & alive_units_image
        new_points_found_image = jax.lax.cond(
            (points_gained == N) & (obs.steps % (params.max_steps_in_match + 1) != 0),
            lambda: jnp.where((farming_units_image==0) & (alive_units_image >0), -1, new_points_found_image),
            lambda: new_points_found_image,
        )

        ########## SYMMETRIZE  ##########
        
        new_points_found_image = symmetrize(team_id, new_points_found_image)

        return RelicPointMemoryState(
            relics_found_image=jax.lax.stop_gradient(new_relics_found_image),
            relics_found_mask=new_relics_found_mask,
            relics_found_positions=new_relics_found_positions,
            points_found_image=jax.lax.stop_gradient(new_points_found_image),
            last_step_team_points=obs.team_points[team_id],
            points_gained=points_gained,
            last_visits_timestep=jax.lax.stop_gradient(new_last_visits_timestep),
        )
