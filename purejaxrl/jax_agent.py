import numpy as np
from agents.base_agent import Agent, N_Actions, N_Agents
from agents.obs import Obs
from luxai_s3.state import EnvParams
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import chex
from purejaxrl.utils import sample_action
from purejaxrl.network import HybridActorCritic
from purejaxrl.make_env import make_env, HybridTransformObs, SimplerActionNoSap
from purejaxrl.utils import init_network_params

def symetric_action_not_vectorized(action: int):
    return [0, 3, 4, 1, 2][action]

class JaxAgent(Agent):
    def __init__(
        self,
        player: str,
        env_params: EnvParams,
    ):
        super().__init__(player, env_params, memory = None)
        env = make_env()
        self.key = jax.random.PRNGKey(0)
        self.network = HybridActorCritic(action_dim=6,)
        self.network_params = init_network_params(self.key, self.network, env)
        self.transform_obs = HybridTransformObs()
        self.transform_action = SimplerActionNoSap()

    @partial(jax.jit, static_argnums=(0,))
    def forward(
        self, 
        key: chex.PRNGKey,
        transformed_obs: Any,
    ):
        transformed_obs_batched = {feat: jnp.expand_dims(value, axis=0) for feat, value in transformed_obs.items()}
        logits, value = self.network.apply(self.network_params, **transformed_obs_batched) # probs is (16, 6)
        return logits 

    def _actions(
        self, 
        obs: Obs,
        remainingOverageTime: int = 60
    ):
        transformed_obs = self.transform_obs.convert(team_id_str=self.player, obs = obs, params=EnvParams.from_dict(self.env_params), reward = 0) 
        logits = self.forward(self.key, transformed_obs=transformed_obs)
        action = sample_action(self.key, logits)[0]
        transformed_action = self.transform_action.convert(team_id_str=self.player, action = action, obs = obs, params=EnvParams.from_dict(self.env_params))
        return transformed_action

    def actions(
        self, obs: Obs, remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]:
        self.update_obs(obs)
        return self._actions(self.expand_obs(obs), remainingOverageTime)

    def act(
        self, step: int, obs: dict[str, Any], remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]:
        return self.actions(Obs.from_dict(obs), remainingOverageTime)
