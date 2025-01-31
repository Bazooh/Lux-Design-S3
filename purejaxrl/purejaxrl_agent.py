import numpy as np
from luxai_s3.state import EnvParams, EnvObs
from typing import Any
import jax, chex
import jax.numpy as jnp
from functools import partial
from purejaxrl.utils import sample_group_action
from purejaxrl.parse_config import parse_config
from purejaxrl.base_agent import JaxAgent
# misc
from purejaxrl.env.memory import Memory
from purejaxrl.env.transform_obs import TransformObs
from purejaxrl.env.transform_action import TransformAction


class RawPureJaxRLAgent(JaxAgent):
    def __init__(
        self,
        player: str,
        state_dict,
        model,
        transform_obs: TransformObs,
        transform_action: TransformAction,
        memory: Memory,
        env_params = EnvParams().__dict__
    ):
        super().__init__(player, env_params, memory)
        
        self.model = model
        self.state_dict = state_dict
        self.transform_obs = transform_obs
        self.transform_action = transform_action

    @partial(jax.jit, static_argnums=(0,1))
    def forward(
        self,
        team_id,
        key: chex.PRNGKey,
        obs: Any,
        memory_state: Any,
        env_params: EnvParams
    ):
        
        transformed_obs = self.transform_obs.convert(team_id=team_id, obs = obs, params=env_params, memory_state=memory_state) 
        transformed_obs_batched = {feat: jnp.expand_dims(value, axis=0) for feat, value in transformed_obs.items()}
        logits, _, _ = self.model.apply(self.state_dict, **transformed_obs_batched, train = False) # logits is (16, 6)
        action = sample_group_action(key, logits[0])[0]
        transformed_action = self.transform_action.convert(
            team_id=self.team_id,
            action=action,
            obs=obs,
            params=env_params,
        )
        return transformed_action 


class PureJaxRLAgent(RawPureJaxRLAgent):
    def __init__(
        self, player: str, env_params = EnvParams().__dict__, config_path: str ="purejaxrl/jax_config.yaml"
    ):
        jax_config = parse_config(config_path)
        super().__init__(player=player, 
                        env_params = env_params, 
                        transform_action=jax_config["env_args"]["transform_action"],
                        transform_obs=jax_config["env_args"]["transform_obs"],
                        memory=jax_config["env_args"]["memory"], 
                        **jax_config["network"] )