import numpy as np
from luxai_s3.state import EnvParams,EnvObs
from typing import Any
import jax, chex
import jax.numpy as jnp
from functools import partial
from purejaxrl.utils import sample_group_action
from purejaxrl.parse_config import parse_config

# misc
from purejaxrl.env.memory import Memory
from purejaxrl.env.transform_obs import TransformObs
from purejaxrl.env.transform_action import TransformAction

def symetric_action_not_vectorized(action: int):
    return [0, 3, 4, 1, 2][action]

class RawJaxAgent:
    def __init__(
        self,
        player: str,
        env_cfg,
        network_params,
        model,
        transform_obs: TransformObs,
        transform_action: TransformAction,
        memory: Memory,
    ):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        
        np.random.seed(0)
        self.env_params = env_cfg
        self.key = jax.random.PRNGKey(0)
        self.model = model
        self.network_params = network_params
        self.transform_obs = transform_obs
        self.transform_action = transform_action
        self.memory = memory
        self.memory_state = self.memory.reset()

    @partial(jax.jit, static_argnums=(0,))
    def forward(
        self, 
        key: chex.PRNGKey,
        transformed_obs: Any,
    ):
        transformed_obs_batched = {feat: jnp.expand_dims(value, axis=0) for feat, value in transformed_obs.items()}
        logits, _, _ = self.model.apply({"params": self.network_params}, **transformed_obs_batched) # logits is (16, 6)
        action = sample_group_action(key, logits[0])
        return action 


    def actions(
        self,
        obs: EnvObs,
        remainingOverageTime: int = 60
    ):
        self.memory_state = self.memory.update(obs = obs, team_id=self.team_id, memory_state=self.memory_state)
        transformed_obs = self.transform_obs.convert(team_id=self.team_id, obs = obs, params=EnvParams.from_dict(self.env_params), memory_state=self.memory_state) 
        action = self.forward(self.key, transformed_obs=transformed_obs)
        transformed_action = self.transform_action.convert(team_id = self.team_id, action = action, obs = transformed_obs, params=EnvParams.from_dict(self.env_params))
        return transformed_action
    

    def act(
        self, 
        step: int, 
        obs: dict[str, Any], 
        remainingOverageTime: int = 60
    ):
        return(self.actions(EnvObs.from_dict(obs)))
    


class JaxAgent(RawJaxAgent):
    def __init__(self, player: str, env_cfg: str, config_path = "purejaxrl/jax_config.yaml"):
        jax_config = parse_config()
        super().__init__(player, env_cfg, 
                        transform_action=jax_config["env_args"]["transform_action"],
                        transform_obs=jax_config["env_args"]["transform_obs"],
                        memory=jax_config["env_args"]["memory"], 
                        **jax_config["network"] )