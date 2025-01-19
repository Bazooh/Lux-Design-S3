from abc import ABC, abstractmethod
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from src.luxai_s3.state import EnvObs
import jax, chex
from functools import partial
from luxai_s3.params import EnvParams, env_params_ranges
from typing import Any
from purejaxrl.env.memory import RelicPointMemoryState
from flax import struct
import jax.numpy as jnp
class Tracker(ABC):
    """
    Abstract base collecting player statistics
    """

    def __init__(self):
        pass


    @abstractmethod
    def get_player_statistics(
        self,
        team_id: int,
        obs: EnvObs,
        mem_state: Any,
        last_obs: EnvObs,
        last_mem_state: Any,
        params: EnvParams,
    ):
        """
        Get player statistics from memory + state
        """
        pass


@struct.dataclass
class PlayerStats:
    points_gained: chex.Array = jnp.array(0, dtype=jnp.int32)
    relics_discovered: chex.Array = jnp.array(0, dtype=jnp.int32)
    points_discovered: chex.Array = jnp.array(0, dtype=jnp.int32)
    energy_gained: chex.Array = jnp.array(0, dtype=jnp.int32)
    ratio_moved: chex.Array = jnp.array(0.0, dtype=jnp.float32)

class GlobalTracker(Tracker):
    def __init__(self):
        super().__init__()
        self.keys = ["points_gained", "cells_dicovered", "relics_discovered", "energy_gained", "points_discovered", "ratio_moved"]

    @partial(jax.jit, static_argnums=(0, 1))
    def get_player_statistics(
        self,
        team_id: int,
        obs: EnvObs,
        mem_state: RelicPointMemoryState,
        last_obs: EnvObs,
        last_mem_state: RelicPointMemoryState,
        params: EnvParams,
    ) -> PlayerStats:
        
        points_gained = mem_state.points_gained

        relics_discovered = jax.numpy.sum((mem_state.relics_found == 1) & (last_mem_state.relics_found == 0))
        points_discovered = jax.numpy.sum((mem_state.points_awarding == 1) & (last_mem_state.points_awarding == 0))
        
        cumulated_energy = jax.numpy.sum(jax.numpy.zeros((24,24), dtype=jax.numpy.int32).at[
            obs.units.position[1- team_id, :, 0],
            obs.units.position[1- team_id, :, 1],
        ].set(obs.units.energy[1 -team_id] + 1))
        last_cumulated_energy = jax.numpy.sum(jax.numpy.zeros((24,24), dtype=jax.numpy.int32).at[
            last_obs.units.position[1- team_id, :, 0],
            last_obs.units.position[1- team_id, :, 1],
        ].set(obs.units.energy[1 -team_id] + 1))
         
        energy_gained = jax.lax.cond(obs.steps%params.max_steps_in_match == 0, 
                                     lambda: cumulated_energy - last_cumulated_energy, 
                                     lambda: 0)
        
        current_positions = obs.units.position[team_id]
        last_positions = last_obs.units.position[team_id]
        active_units = obs.units_mask[team_id]
        moved_units = jax.numpy.any(current_positions != last_positions, axis=-1) & active_units
        ratio_moved = jax.numpy.sum(moved_units) / jax.numpy.sum(active_units)

        return PlayerStats(
            points_gained=points_gained,
            relics_discovered=relics_discovered,
            points_discovered=points_discovered,
            energy_gained=energy_gained,
            ratio_moved=ratio_moved,
        )

