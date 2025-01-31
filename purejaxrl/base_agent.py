import numpy as np
from luxai_s3.state import EnvParams, EnvObs
from typing import Any
import jax, chex
import jax.numpy as jnp
from functools import partial
from abc import ABC, abstractmethod
from purejaxrl.env.memory import Memory

class JaxAgent(ABC):
    def __init__(
        self,
        player: str,
        env_params,
        memory: Memory        
    ):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_params = EnvParams.from_dict(env_params)
        self.key = jax.random.PRNGKey(0)
        
        self.memory = memory        
        self.memory_state = self.memory.reset()

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,1))
    def forward(
        self,
        team_id,
        key: chex.PRNGKey,
        obs: Any,
        memory_state: Any,
        env_params: EnvParams
    ):
        ...

    def actions(
        self,
        obs: EnvObs,
        remainingOverageTime: int = 60
    ):
        new_key, key = jax.random.split(self.key)
        self.key = new_key
        self.memory_state = self.memory.update(obs = obs, team_id=self.team_id, memory_state=self.memory_state, params = self.env_params)
        action = self.forward(team_id = self.team_id, key = key, obs = obs, memory_state = self.memory_state, env_params = self.env_params)
        return action

    def act(self, step: int, obs: dict[str, Any], remainingOverageTime: int = 60):
        return self.actions(EnvObs.from_dict(obs))
