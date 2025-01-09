import jax
import chex
from functools import partial
from typing import Optional, Tuple, Union, Any
from purejaxrl.wrappers.base_wrappers import GymnaxWrapper, EnvState, EnvParams
from purejaxrl.wrappers.transform_reward import TransformReward

class TransformRewardWrapper(GymnaxWrapper):
    def __init__(self, env, transform_reward: TransformReward):
        super().__init__(env)
        self.transform_reward = transform_reward

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        env_state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, env_state, action, params)
        transformed_reward = {k: self.transform_reward.convert(team_id_str=k, obs = obs[k], params = params, reward = r) for k, r in reward.items()}
        return obs, state, transformed_reward, done, info

