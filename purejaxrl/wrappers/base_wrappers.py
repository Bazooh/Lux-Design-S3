import jax
import chex
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment
from luxai_s3.env import LuxAIS3Env, EnvObs, EnvState, EnvParams

from sample_params import sample_params, sample_params_fn

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

@struct.dataclass
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    timestep: int

class SimplifyTruncation(GymnaxWrapper):
    """"
    Wraps the env from the format:
        obs, state, reward, terminated_dict, truncated_dict, info
        to
        obs, env_state, reward, done, info 
    """

    def __init__(self, env: LuxAIS3Env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        env_state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, env_state, reward, terminated_dict, truncated_dict, info = self._env.step(
            key, env_state, action, params
        )
        done = truncated_dict["player_0"] | terminated_dict["player_0"]
        return obs, env_state, reward, done, info 
    

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""
    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, 
        params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, params)
        log_env_state = LogEnvState(
                                    env_state=env_state, 
                                    episode_returns = {k: 0 for k in obs.keys()},
                                    episode_lengths=0,
                                    timestep=0
                                )
        
        return obs, log_env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        log_env_state: LogEnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, log_env_state.env_state, action, params
        )
        with jax.numpy_dtype_promotion('standard'): 
            new_episode_return = {k: (log_env_state.episode_returns[k] + r)* (1 - done) for (k,r) in reward.items()}
            new_episode_length = (log_env_state.episode_lengths + 1) * (1 - done)
            new_log_env_state = LogEnvState(
                env_state=env_state,
                episode_returns = new_episode_return,
                episode_lengths = new_episode_length,
                timestep=log_env_state.timestep+1
            )
        info["timestep"] = new_log_env_state.timestep
        info["returned_episode"] = done
        return obs, new_log_env_state, reward, done, info
