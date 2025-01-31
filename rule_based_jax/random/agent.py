from purejaxrl.base_agent import JaxAgent
import jax, chex
from functools import partial
from typing import Any
from luxai_s3.env import EnvObs, EnvParams
import jax.numpy as jnp
from purejaxrl.env.memory import Memory, NoMemory
from rule_based_jax.utils import find_nearest, direction_to

class RandomAgent_Jax(JaxAgent):
    def __init__(self, player: str, env_params = EnvParams().__dict__):
        super().__init__(player, env_params, memory = NoMemory())

    
    @partial(jax.jit, static_argnums=(0,1))
    def forward(
        self,
        team_id,
        key: chex.PRNGKey,
        obs: EnvObs,
        memory_state: Any,
        env_params: EnvParams
    ):
        positions = obs.units.position[team_id]
        
        key_group = jax.random.split(key, 16)
        random_direction = lambda pos, key: direction_to(src = pos, target = jax.random.randint(key, shape=(2), minval=0, maxval=23))

        @jax.jit
        def get_ship_move_action(key, idx): 
            pos = positions[idx]
            action = random_direction(pos, key)
            return action

        move_actions = jax.vmap(get_ship_move_action)(key_group, jnp.arange(16))
        action = jnp.zeros((16,3), dtype=jnp.int32)
        action = action.at[:,0].set(move_actions)
        
        return action