import jax
import chex
from functools import partial
from typing import Optional, Tuple, Union, Any
from purejaxrl.wrappers.base_wrappers import GymnaxWrapper, EnvState, EnvParams
from purejaxrl.wrappers.transform_obs import TransformObs


class TransformObservation(GymnaxWrapper):
    def __init__(self, env, transform_obs: TransformObs):
        super().__init__(env)
        self.transform_obs = transform_obs
        self.observation_shape = self.transform_obs.observation_shape
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        transformed_obs = ({k: self.transform_obs.convert(team_id_str=k, obs = o, params = params, reward= None) for k, o in obs.items()})
        return transformed_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        env_state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, env_state, action, params)
        transformed_obs = ({k: self.transform_obs.convert(team_id_str=k, obs = o, params = params, reward= None) for k, o in obs.items()})
        return transformed_obs, state, reward, done, info
