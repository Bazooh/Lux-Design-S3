from typing import Any
from luxai_s3.state import EnvObs, EnvParams
import jax
import chex
import jax.numpy as jnp
from functools import partial
from purejaxrl.base_agent import JaxAgent
from purejaxrl.env.memory import Memory, RelicPointMemory, RelicPointMemoryState
from rule_based_jax.utils import find_nearest, direction_to
import jax.scipy.signal
from purejaxrl.env.transform_action import SimplerActionWithSap
import numpy as np
import os

RING_EXPLO = 3
RING_POINT = 7
RING_RELIC = 13


class NaiveAgent_Jax(JaxAgent):
    def __init__(self, player: str, env_params=EnvParams().__dict__):
        super().__init__(player, env_params, memory=RelicPointMemory())
        self.transform_action = SimplerActionWithSap(do_mirror_input=False)

    @partial(jax.jit, static_argnums=(0, 1))
    def forward(
        self,
        team_id: int,
        key: chex.PRNGKey,
        obs: EnvObs,
        memory_state: RelicPointMemoryState,
        env_params: EnvParams,
    ):
        # Extract information from memory and observations
        relics_found = memory_state.relics_found_image
        points_found = memory_state.points_found_image
        positions = obs.units.position[team_id]

        alive_units_image = jnp.zeros((24, 24), dtype=jnp.int8)
        alive_units_image = alive_units_image.at[positions[:, 0], positions[:, 1]].set(
            1
        )
        alive_units_image = (alive_units_image == 1)

        point_not_occupied_by_allies = (points_found == 1) & (alive_units_image < 1)

        # Create convolved images for nearby points, relics, and unseen areas
        points_near_image = (
            jax.scipy.signal.convolve2d(
                point_not_occupied_by_allies.astype(jnp.int8),
                jnp.ones((RING_POINT, RING_POINT)),
                mode="same",
            )
            > 0
        )
        padded_points_not_occupied_by_allies = jnp.pad(
            point_not_occupied_by_allies, ((RING_POINT // 2, RING_POINT // 2), (RING_POINT // 2, RING_POINT // 2))
        )
        
        relic_near_image = (
            jax.scipy.signal.convolve2d(
                (relics_found == 1).astype(jnp.int8), jnp.ones((5, 5)), mode="same"
            )
            > 0
        )
        relic_near_image_large_ring = (
            jax.scipy.signal.convolve2d(
                (relics_found == 1).astype(jnp.int8),
                jnp.ones((RING_RELIC, RING_RELIC)),
                mode="same",
            )
            > 0
        )

        # Generate random keys for each unit
        key_group = jax.random.split(key, positions.shape[0])
        padded_relics_found = jnp.pad(
            relics_found, ((RING_RELIC // 2, RING_RELIC // 2), (RING_RELIC // 2, RING_RELIC // 2))
        )
        go_to_relic = lambda pos, key: find_nearest(
            key,
            jax.lax.dynamic_slice(
                padded_relics_found == 1,    
                (pos[0], pos[1]),
                (RING_RELIC, RING_RELIC),
            ),
            RING_RELIC,
        )
        go_to_point = lambda pos, key: find_nearest(
            key,
            jax.lax.dynamic_slice(
                padded_points_not_occupied_by_allies,
                (pos[0], pos[1]),
                (RING_POINT, RING_POINT),
            )
        )
        random_direction = lambda pos, key: direction_to(
            src=pos, target=jax.random.randint(key, shape=(2), minval=0, maxval=23)
        )

        random_exploration = lambda pos, key, idx: jax.random.choice(
            key,
            a=jnp.arange(5),
            p=jnp.array([0.0, 0.0, 0.5 + 0.2 * (((idx + (2*idx)) % 4) - 1.5), 0.5 - 0.2 *  (((idx + (2*idx)) % 4) - 1.5), 0.0])
            if team_id == 0
            else jnp.array([0.0, 0.5 + 0.2 * (((idx + (2*idx)) % 4) - 1.5), 0.0, 0.0, 0.5 - 0.2 * (((idx + (2*idx)) % 4) - 1.5)]),
        )

        # Define a function to compute actions for each ship
        def get_ship_move_action(key, idx):
            pos = positions[idx]
            draw = jax.random.uniform(key)
            # Check for nearby points, relics, or unseen areas
            action = jax.lax.select(
                obs.steps % 101
                < 30
                | ~((relics_found == 1).any()),
                random_exploration(pos, key, idx),
                jax.lax.select(
                    relic_near_image_large_ring[pos[0], pos[1]] & ~relic_near_image[pos[0], pos[1]],
                    go_to_relic(pos, key),
                    jax.lax.select(
                        relic_near_image[pos[0], pos[1]],
                        jax.lax.select(
                            draw < 0.33,
                            go_to_relic(pos, key),
                            random_direction(pos, key),
                        ),
                        random_direction(pos, key),
                    ),
                ),
            )
            return action #int

        # Vectorize the action computation
        move_actions = jax.vmap(get_ship_move_action)(
            key_group, jnp.arange(positions.shape[0])
        )

        # Construct the action array
        action = jnp.zeros((positions.shape[0], 3), dtype=jnp.int32)
        action = action.at[:, 0].set(move_actions)

        sapactions = jnp.ones_like(action[:, 0]) * 5
        auxaction = jax.lax.select(obs.steps%101 < 95, action[:,0], sapactions)
        # jax.debug.print("team id : {team_id}, step : {step}, action : {action}", team_id = team_id, step = obs.steps, action = move_actions)
        action = self.transform_action.convert(team_id, auxaction, obs, env_params)

        return action
