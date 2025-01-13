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
    points_gained: int

class RelicPointMemory(Memory):
    def __init__(self):
        pass

    def reset(self) -> RelicPointMemoryState:
        return RelicPointMemoryState(
                    relics_found=jnp.zeros((24,24), dtype = jnp.int8),
                    points_awarding= jnp.zeros((24,24), dtype = jnp.int8),
                    last_step_team_points=0,
                    points_gained=0
        )
    #@partial(jax.jit, static_argnums=(0,2))
    def update(self, obs: EnvObs, team_id: int, memory_state: RelicPointMemoryState):
        """
        Update rule 1:
        - Update the relics_found to 1 if a relic is discovered.
        Update rule 2:
        - If I expect N points this round (based on the point map), and I win exactly N points, then set the useless ones to -1
        Update rule 3:
        - If a unit is oob of relic, then set the points_awarding of the units position to -1.
        Update rule 4:
        - If N is the number of the units in range of a relic, and we have gained N points, then set the points_awarding of all the units positions to 1.
        """

        def compute_farming_units(units_pos, points_awarding):
            """
            Identify the units that are farming relics. A unit is farming if it sits on a points awarding square.
            """
            def is_farming(unit_pos):
                return points_awarding[unit_pos[0], unit_pos[1]] == 1
            return jax.vmap(is_farming)(units_pos)
        
        def compute_inbound_units(units_pos, new_relics_found):
            """
            Identify units that are not within a Manhattan distance of 2 from any relic.
            """
            def is_inbound(unit_pos):
                neighbors = jnp.array([
                        [unit_pos[0] + i, unit_pos[1] + j]
                        for i in range(-2, 3)
                        for j in range(-2, 3)
                ])
                
                # Ensure neighbors are within bounds
                valid_mask = ((neighbors >= 0) & (neighbors < 24)).all(axis=1)  # Check both x and y are within bounds
                neighbors = jnp.where(valid_mask[:, None], neighbors, -1)  # Replace invalid neighbors with -1
                
                # Mask out invalid neighbors before indexing
                neighbor_values = jnp.where(
                    (neighbors[:, 0] >= 0) & (neighbors[:, 1] >= 0),  # Only consider valid neighbors
                    new_relics_found[neighbors[:, 0], neighbors[:, 1]],
                    0  # Default to 0 for invalid neighbors
                )
                return jnp.any(neighbor_values == 1)
            return jax.vmap(is_inbound)(units_pos)
        
        def compute_oob_units(units_pos, new_relics_found):
            """
            Identify units that are not within a Manhattan distance of 2 from any relic.
            """
            def is_inbound(unit_pos):
                neighbors = jnp.array([
                        [unit_pos[0] + i, unit_pos[1] + j]
                        for i in range(-2, 3)
                        for j in range(-2, 3)
                ])
                
                # Ensure neighbors are within bounds
                valid_mask = ((neighbors >= 0) & (neighbors < 24)).all(axis=1)  # Check both x and y are within bounds
                neighbors = jnp.where(valid_mask[:, None], neighbors, -1)  # Replace invalid neighbors with -1
                
                # Mask out invalid neighbors before indexing
                neighbor_values = jnp.where(
                    (neighbors[:, 0] >= 0) & (neighbors[:, 1] >= 0),  # Only consider valid neighbors
                    new_relics_found[neighbors[:, 0], neighbors[:, 1]],
                    0  # Default to 0 for invalid neighbors
                )
                return jnp.all(neighbor_values < 1)
            return jax.vmap(is_inbound)(units_pos)
    
        
        ########## Update rule 1:  ##########
        new_relics_found = memory_state.relics_found
        new_points_awarding = memory_state.points_awarding
        viewed_relic_obs_indices = obs.get_avaible_relics()
        currently_viewing_relics = jnp.zeros((24,24), dtype = jnp.int8).at[obs.relic_nodes[viewed_relic_obs_indices, 0], obs.relic_nodes[viewed_relic_obs_indices, 1]].set(1)
        new_relics_found = new_relics_found + obs.sensor_mask * (
            (new_relics_found == 0) * (2 * currently_viewing_relics - 1)
        )

        ########## Update rule 2:  ##########
        points_gained = obs.team_points[team_id] - memory_state.last_step_team_points

        alive_units_mask = obs.units_mask[team_id]
        alive_units_pos = obs.units.position[team_id][alive_units_mask]

        farming_units_mask = compute_farming_units(obs.units.position[team_id], new_points_awarding)
        not_farming_a_priori_pos = obs.units.position[team_id][~farming_units_mask]

        outbound_units_mask = compute_inbound_units(obs.units.position[team_id], new_relics_found)
        outbound_units_pos = obs.units.position[team_id][outbound_units_mask]

        inbounds_units_mask = compute_oob_units(obs.units.position[team_id], new_relics_found)
        inbounds_units_pos = obs.units.position[team_id][inbounds_units_mask]
    
        new_points_awarding = jax.lax.cond(
            points_gained == jnp.sum(farming_units_mask),
            lambda _: new_points_awarding.at[not_farming_a_priori_pos[:, 0], not_farming_a_priori_pos[:, 1]].set(-1),
            lambda _: new_points_awarding,
            operand=None
        )

        ########## Update rule 3:  ##########
        new_points_awarding = new_points_awarding.at[outbound_units_pos[:, 0], outbound_units_pos[:, 1]].set(-1)

        ########## Update rule 4:  ##########
        new_points_awarding = jax.lax.cond(
            points_gained == jnp.sum(inbounds_units_mask),
            lambda _: new_points_awarding.at[inbounds_units_pos[:, 0], inbounds_units_pos[:, 1]].set(1),
            lambda _: new_points_awarding,
            operand=None
        )

        ########## Update rule 5:  ##########
        new_points_awarding = jax.lax.cond(
            points_gained == 0,
            lambda _: new_points_awarding.at[alive_units_pos[:, 0], alive_units_pos[:, 1]].set(-1),
            lambda _: new_points_awarding,
            operand=None
        )

        return RelicPointMemoryState(
            relics_found=new_relics_found,
            points_awarding=new_points_awarding,
            last_step_team_points=obs.team_points[team_id],
            points_gained=points_gained
        )
    
    def expand(self, obs, team_id, memory_state):
        return obs