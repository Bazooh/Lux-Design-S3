import jax
import chex
from functools import partial
from typing import Optional, Tuple, Union, Any
from purejaxrl.wrappers.base_wrappers import GymnaxWrapper, EnvState, EnvParams

class TransformReward(GymnaxWrapper):
    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        env_state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, env_state, action, params)
        transformed_r = ({k: self.transform_reward(r) for k, r in obs.items()})
        return obs, state, transformed_r, done, info

