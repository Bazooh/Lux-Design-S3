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
from purejaxrl.wrappers.transform_reward import TransformReward
from purejaxrl.wrappers.transform_obs import TransformObs
from purejaxrl.wrappers.transform_action import TransformAction
SimplerPlayerAction = np.ndarray[Literal[16], np.dtype[np.int32]]

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env: LuxAIS3Env):
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
        action: PlayerAction,
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
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, log_env_state.env_state, action, params
        )
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

class TransformRewardWrapper(GymnaxWrapper):
    def __init__(self, env, transform_reward: TransformReward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(
        self,
        key: chex.PRNGKey,
        env_state: EnvState,
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, env_state, action, params)
        transformed_reward = {k: self.transform_reward.convert(team_id_str=k, obs = obs[k], params = params, reward = r) for k, r in reward.items()}
        return obs, state, transformed_reward, done, info


class TransformObsWrapper(GymnaxWrapper):
    def __init__(self, env, transform_obs: TransformObs):
        super().__init__(env)
        self.transform_obs = transform_obs
        self.observation_space = self.transform_obs.observation_space
    
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        transformed_obs = {k: self.transform_obs.convert(team_id_str=k, obs = o, params = params, reward = 0) for k, o in obs.items()}
        return transformed_obs, state

    def step(
        self,
        key: chex.PRNGKey,
        env_state: EnvState,
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, env_state, action, params)
        transformed_obs = {k: self.transform_obs.convert(team_id_str=k, obs = o, params = params, reward = reward) for k, o in obs.items()}
        return transformed_obs, state, reward, done, info

class TransformActionWrapper(GymnaxWrapper):
    """"
    Wraps the env from the format:
        obs, state, reward, terminated_dict, truncated_dict, info
        to
        obs, env_state, reward, done, info 
    """

    def __init__(self, env: LuxAIS3Env, transform_action: TransformAction):
        self.transform_action = transform_action
        self.action_space = transform_action.action_space
        super().__init__(env)

    def step(
        self,
        key: chex.PRNGKey,
        env_state: EnvState,
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        last_obs = self._env.get_obs(env_state)
        action = {k: self.transform_action.convert(team_id_str=k, action = a, obs = last_obs[k], params = params) for k, a in action.items()}
        obs, env_state, reward, done, info = self._env.step(
            key, env_state, action, params
        )
        return obs, env_state, reward, done, info 