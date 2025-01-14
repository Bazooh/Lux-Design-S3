import gymnax.environments.spaces
import jax
import chex
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any, Literal
import numpy as np 
import jax.numpy as jnp
from gymnax.environments import environment
import gymnax
from luxai_s3.env import LuxAIS3Env, EnvState, EnvParams, PlayerAction
from purejaxrl.wrappers.transform_reward import TransformReward
from purejaxrl.wrappers.transform_obs import TransformObs
from purejaxrl.wrappers.transform_action import TransformAction
from purejaxrl.wrappers.memory import Memory
from purejaxrl.wrappers.symmetry import Symmetry
SimplerPlayerAction = np.ndarray[Literal[16], np.dtype[np.int32]]

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env: LuxAIS3Env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


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

@struct.dataclass
class Env_Mem_State:
    env_state: EnvState
    memory_state_player_0: Any
    memory_state_player_1: Any

class MemoryWrapper(GymnaxWrapper):
    def __init__(self, env: LuxAIS3Env, memory: Memory):
        super().__init__(env)
        self.memory = memory

    def reset(self, key: chex.PRNGKey, params: Optional[EnvParams] = None) -> Tuple[chex.Array, Env_Mem_State]:
        obs, env_state = self._env.reset(key, params)
        memory_state_player_0 = self.memory.reset()
        memory_state_player_1 = self.memory.reset()
        env_mem_state = Env_Mem_State(
                env_state=env_state, 
                memory_state_player_0 = memory_state_player_0, 
                memory_state_player_1 = memory_state_player_1        
        )
        return obs, env_mem_state

    def step(
        self,
        key: chex.PRNGKey,
        env_mem_state: Env_Mem_State,
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, Env_Mem_State, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(key, env_mem_state.env_state, action, params)
        memory_state_player_0 = self.memory.update(obs['player_0'], 0, env_mem_state.memory_state_player_0)
        memory_state_player_1 = self.memory.update(obs['player_1'], 1, env_mem_state.memory_state_player_1)
        env_mem_state = Env_Mem_State(
                env_state=env_state, 
                memory_state_player_0 = memory_state_player_0, 
                memory_state_player_1 = memory_state_player_1        
        )
        expanded_obs = {
            'player_0': self.memory.expand(obs['player_0'], 0, env_mem_state.memory_state_player_0),
            'player_1': self.memory.expand(obs['player_1'], 1, env_mem_state.memory_state_player_1)
        }
        return expanded_obs, env_mem_state, reward, done, info

@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_return: list[float]
    episode_points: list[int]
    episode_wins: list[int]
    episode_timestep: int
    global_timestep: int

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
            episode_return=jnp.array([0, 0]),
            episode_points=jnp.array([0, 0]),
            episode_wins=jnp.array([0, 0]),
            episode_timestep=0,
            global_timestep=0
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

        new_episode_return = (log_env_state.episode_return + jnp.array([reward["player_0"], reward["player_1"]])) * (1 - done)
        new_episode_points =  info["final_observation"]["player_0"].team_points * (1 - done)
        new_episode_wins = info["final_observation"]["player_0"].team_wins * (1 - done)
        new_log_env_state = LogEnvState(
            env_state=env_state,
            episode_return=new_episode_return,
            episode_points=new_episode_points,
            episode_wins=new_episode_wins,
            episode_timestep=(log_env_state.episode_timestep + 1)*(1 - done),
            global_timestep=log_env_state.global_timestep + 1,
        )
        info["episode_return"] = new_log_env_state.episode_return
        info["episode_points"] = new_log_env_state.episode_points
        info["episode_wins"] = new_log_env_state.episode_wins
        info["global_timestep"] = new_log_env_state.global_timestep
        info["episode_timestep"] = new_log_env_state.episode_timestep
        info["returned_episode"] = done
        return obs, new_log_env_state, reward, done, info

class TransformRewardWrapper(GymnaxWrapper):
    def __init__(self, env, transform_reward: TransformReward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(
        self,
        key: chex.PRNGKey,
        env_mem_state: Env_Mem_State,
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, Env_Mem_State, float, bool, dict]:
        last_obs = self._env.get_obs(env_mem_state.env_state)
        obs, state, reward, done, info = self._env.step(key, env_mem_state, action, params)
        transformed_reward = {
            "player_0": self.transform_reward.convert(team_id=0, last_obs=last_obs["player_0"], obs = obs["player_0"], reward = reward["player_0"], params = params),
            "player_1": self.transform_reward.convert(team_id=1, last_obs=last_obs["player_1"], obs = obs["player_1"], reward = reward["player_1"], params = params),
        }
        return obs, state, transformed_reward, done, info


class TransformObsWrapper(GymnaxWrapper):
    def __init__(self, env, transform_obs: TransformObs):
        super().__init__(env)
        self.transform_obs = transform_obs
        self.observation_space = self.transform_obs.observation_space
    
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        transformed_obs = {
            "player_0": self.transform_obs.convert(team_id=0, obs = obs["player_0"], params = params, memory_state=state.memory_state_player_0),
            "player_1": self.transform_obs.convert(team_id=1, obs = obs["player_1"], params = params, memory_state=state.memory_state_player_1),
        }
        return transformed_obs, state

    def step(
        self,
        key: chex.PRNGKey,
        env_mem_state: Env_Mem_State,
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, Env_Mem_State, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, env_mem_state, action, params)
        transformed_obs = {
            "player_0": self.transform_obs.convert(team_id=0, obs = obs["player_0"], params = params, memory_state=state.memory_state_player_0),
            "player_1": self.transform_obs.convert(team_id=1, obs = obs["player_1"], params = params, memory_state=state.memory_state_player_1),
        }
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
        env_mem_state: Env_Mem_State,
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, Env_Mem_State, float, bool, dict]:
        current_obs = self._env.get_obs(env_mem_state.env_state)
        action = {
            "player_0": self.transform_action.convert(team_id=0, action=action["player_0"], obs = current_obs["player_0"], params = params),
            "player_1": self.transform_action.convert(team_id=1, action=action["player_1"], obs = current_obs["player_1"], params = params),
        }
        obs, env_state, reward, done, info = self._env.step(
            key, env_mem_state, action, params
        )
        return obs, env_state, reward, done, info 
  
class SymmetryWrapper(GymnaxWrapper):
    def __init__(self, env: LuxAIS3Env, symmetry: Symmetry):
        super().__init__(env)
        self.symmetry = symmetry

    def reset(self, key: chex.PRNGKey, params: Optional[EnvParams] = None):
        obs, state = self._env.reset(key, params)
        symmetrized_obs = {
            "player_0": self.symmetry.convert_obs(team_id=0, obs = obs["player_0"]),
            "player_1": self.symmetry.convert_obs(team_id=1, obs = obs["player_1"]),
        }
        return symmetrized_obs, state

    def step(
        self,
        key: chex.PRNGKey,
        env_mem_state: Env_Mem_State,
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, Env_Mem_State, float, bool, dict]:
        print(action["player_0"].shape)
        unsymmetrized_action = {
            "player_0": self.symmetry.convert_action(team_id=0, action = action["player_0"]),
            "player_1": self.symmetry.convert_action(team_id=1, action = action["player_1"]),
        } # turns a symmetric action back to an unsymmetric one
        obs, state, reward, done, info = self._env.step(key, env_mem_state, unsymmetrized_action, params)
        symmetrized_obs = {
            "player_0": self.symmetry.convert_obs(team_id=0, obs = obs["player_0"]),
            "player_1": self.symmetry.convert_obs(team_id=1, obs = obs["player_1"]),
        }
        return symmetrized_obs, state, reward, done, info
    
