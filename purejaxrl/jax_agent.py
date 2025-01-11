from luxai_s3.state import EnvParams, EnvObs
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import chex
from purejaxrl.wrappers.transform_obs import TransformObs
from purejaxrl.wrappers.transform_action import TransformAction
from purejaxrl.utils import sample_action
from purejaxrl.network import HybridActorCritic
from purejaxrl.make_env import make_env, HybridTransformObs, SimplerActionNoSap
from purejaxrl.utils import init_network_params

def symetric_action_not_vectorized(action: int):
    return [0, 3, 4, 1, 2][action]

class JaxAgent:
    def __init__(
        self,
        player: str,
        env_params: EnvParams,
    ):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_params = env_params # dictionnary form !
        
        self.key = jax.random.PRNGKey(0)
        env = make_env()
        self.network = HybridActorCritic(action_dim=env.action_space.n,)
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
        obs: EnvObs,
        remainingOverageTime: int = 60
    ):
        transformed_obs = self.transform_obs.convert(team_id_str=self.player, obs = obs, params=EnvParams.from_dict(self.env_params), reward = 0) 
        logits = self.forward(self.key, transformed_obs=transformed_obs)
        action = sample_action(self.key, logits)[0]
        transformed_action = self.transform_action.convert(team_id_str=self.player, action = action, obs = obs, params=EnvParams.from_dict(self.env_params))
        return transformed_action

    def _default_actions(
        self, 
        obs: EnvObs,
        remainingOverageTime: int = 60
    ):
        actions = np.zeros((16, 3), dtype=np.int32)
        return actions
    
    def actions(
        self, obs: EnvObs, remainingOverageTime: int = 60
    ):
        return self._default_actions(obs, remainingOverageTime)

    def act(
        self, step: int, obs: dict[str, Any], remainingOverageTime: int = 60
    ):
        return self.actions(EnvObs.from_dict(obs), remainingOverageTime)
