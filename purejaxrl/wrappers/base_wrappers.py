import gymnax.environments.spaces
import jax
import chex
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any, Literal
import numpy as np 
from gymnax.environments import environment
import gymnax
from luxai_s3.env import LuxAIS3Env, EnvState, EnvParams, PlayerAction

SimplerPlayerAction = np.ndarray[Literal[16], np.dtype[np.int32]]

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

    def step(
        self,
        key: chex.PRNGKey,
        env_state: EnvState,
        action: PlayerAction | SimplerPlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, env_state, reward, terminated_dict, truncated_dict, info = self._env.step(
            key, env_state, action, params
        )
        done = truncated_dict["player_0"] | terminated_dict["player_0"]
        return obs, env_state, reward, done, info 

class SimplifyAction(GymnaxWrapper):
    """"
    Wraps the env from the format:
        obs, state, reward, terminated_dict, truncated_dict, info
        to
        obs, env_state, reward, done, info 
    """

    def __init__(self, env: LuxAIS3Env):
        self.action_space = gymnax.environments.spaces.Discrete(6) 
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def infer_action(self, simpler_action: SimplerPlayerAction) -> PlayerAction:
        action = jax.numpy.zeros((16,3), dtype=jax.numpy.int16)
        with jax.numpy_dtype_promotion('standard'): 
            action = action.at[:,0].set(simpler_action)
        return action
        
    def step(
        self,
        key: chex.PRNGKey,
        env_state: EnvState,
        simpler_action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        action = {k: self.infer_action(a) for k, a in simpler_action.items()}
        obs, env_state, reward, done, info = self._env.step(
            key, env_state, action, params
        )
        return obs, env_state, reward, done, info 
    
class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""
    def __init__(self, env: environment.Environment):
        super().__init__(env)

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

    def step(
        self,
        key: chex.PRNGKey,
        log_env_state: LogEnvState,
        action: PlayerAction | SimplerPlayerAction,
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
