import jax
import chex
from functools import partial
from typing import Optional, Tuple, Union, Any
from base_wrappers import GymnaxWrapper, EnvState, EnvParams

class TransformObservation(GymnaxWrapper):
    def __init__(self, env, transform_obs):
        super().__init__(env)
        self.transform_obs = transform_obs
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        env_state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, env_state, action, params)
        return self.transform_obs(obs), state, reward, done, info

