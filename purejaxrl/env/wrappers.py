import flax.struct
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
from purejaxrl.env.transform_reward import TransformReward
from purejaxrl.env.transform_obs import TransformObs
from purejaxrl.env.transform_action import TransformAction
from purejaxrl.env.tracker import Tracker
from purejaxrl.env.memory import Memory
import flax
# for recording
from purejaxrl.utils import serialize_metadata
from luxai_s3.state import serialize_env_actions, serialize_env_states
from luxai_s3.params import serialize_env_params
import os, json
from luxai_runner.episode import json_to_html
from luxai_s3.env import LuxAIS3Env, EnvState, EnvParams, PlayerAction

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env: LuxAIS3Env):
        self._env = env
        self.num_agents = 2
        self.agents = ["player_0", "player_1"]
    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


################################## RECORD WRAPPER ##################################
class RecordEpisodeWrapper(GymnaxWrapper): # adapted from the gym record wrapper
    def __init__(
        self,
        env: LuxAIS3Env,
        save_dir: str | None = None,
        save_on_close: bool = True,
        save_on_reset: bool = True,
        save_format: Literal["json", "html"] = "json",
    ):
        super().__init__(env)
        self.episode = dict(states=[], actions=[], metadata=dict())
        self.episode_id = 0
        self.save_dir = save_dir
        self.save_on_close = save_on_close
        self.save_on_reset = save_on_reset
        self.episode_steps = 0
        self.save_format: Literal["json", "html"] = save_format
        if save_dir is not None:
            from pathlib import Path
            Path(save_dir).mkdir(parents=True, exist_ok=True)

    def reset(
        self, key: chex.PRNGKey, 
        params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        if self.save_on_reset and self.episode_steps > 0:
            self._save_episode_and_reset()
        obs, env_state = self._env.reset(key, params)
        self.episode["metadata"]["seed"] = key
        self.episode["params"] = params
        self.episode["states"].append(env_state)
        return obs, env_state

    def step(
        self,
        key: chex.PRNGKey,
        env_state: EnvState,
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, env_state, reward, terminated, truncated, info = self._env.step(
            key, env_state, action, params
        )
        self.episode_steps += 1
        self.episode["states"].append(info["final_state"])
        self.episode["actions"].append(action)
        return obs, env_state, reward, terminated, truncated, info
    
    def serialize_episode_data(self, episode=None):
        if episode is None:
            episode = self.episode
        ret = dict()
        ret["observations"] = serialize_env_states(episode["states"])
        if "actions" in episode:
            ret["actions"] = serialize_env_actions(episode["actions"])
        ret["metadata"] = serialize_metadata(episode["metadata"])
        ret["params"] = serialize_env_params(episode["params"])
        return ret
    
    def save_episode(self, save_path: str):
        episode = self.serialize_episode_data()
        with open(save_path, "w") as f:
            if self.save_format == "json":
                json.dump(episode, f)
            else:
                f.write(json_to_html(episode))
        self.episode = dict(states=[], actions=[], metadata=dict())

    def _save_episode_and_reset(self):
        """saves to generated path based on self.save_dir and episoe id and updates relevant counters"""
        self.save_episode(
            os.path.join(self.save_dir, f"episode_{self.episode_id}.{self.save_format}")
        )
        self.episode_id += 1
        self.episode_steps = 0
        
    def close(self):
        if self.save_on_close and self.episode_steps > 0:
            self._save_episode_and_reset()


################################## SIMPLIFY TRUNCATION WRAPPER ##################################
class SimplifyTruncationWrapper(GymnaxWrapper):
    """"
    Wraps the env from the format:
        (obs, state, reward, terminated_dict, truncated_dict, info)
        to
        (obs, env_state, reward, done, info)
    """

    def __init__(self, env: LuxAIS3Env | RecordEpisodeWrapper):
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


################################## MEMORY WRAPPER ##################################
@struct.dataclass
class Env_Mem_State:
    env_state: EnvState
    memory_state_player_0: Any
    memory_state_player_1: Any

class MemoryWrapper(GymnaxWrapper):
    def __init__(self, env: SimplifyTruncationWrapper, memory: Memory):
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
        memory_state_player_0 = self.memory.update(obs = obs['player_0'], team_id=0, memory_state=env_mem_state.memory_state_player_0)
        memory_state_player_1 = self.memory.update(obs = obs['player_1'], team_id=1, memory_state=env_mem_state.memory_state_player_1)
        env_mem_state = Env_Mem_State(
                env_state=env_state, 
                memory_state_player_0 = memory_state_player_0, 
                memory_state_player_1 = memory_state_player_1        
        )
        expanded_obs = {
            'player_0': self.memory.expand(obs = obs['player_0'], team_id=0, memory_state=env_mem_state.memory_state_player_0),
            'player_1': self.memory.expand(obs = obs['player_1'], team_id=1, memory_state=env_mem_state.memory_state_player_1)
        }
        return expanded_obs, env_mem_state, reward, done, info


################################## TRACKER WRAPPER ##################################
class TrackerWrapper(GymnaxWrapper):
    def __init__(self, env: MemoryWrapper, tracker: Tracker):
        super().__init__(env)
        self.tracker = tracker

    def step(
        self,
        key: chex.PRNGKey,
        env_state: Env_Mem_State,
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, Env_Mem_State, float, bool, dict]:
        last_mem_state_player_0, last_mem_state_player_1 = env_state.memory_state_player_0, env_state.memory_state_player_1 
        last_obs = self._env.get_obs(env_state.env_state)
        obs, env_state, reward, done, info = self._env.step(key, env_state, action, params)
        info["stats"]={
            "player_0": self.tracker.get_player_statistics(team_id=0, 
                                                    last_obs=last_obs["player_0"], 
                                                    last_mem_state = last_mem_state_player_0, 
                                                    obs = obs["player_0"], 
                                                    mem_state = env_state.memory_state_player_0,
                                                    params = params),
            "player_1": self.tracker.get_player_statistics(team_id=1, 
                                                    last_obs=last_obs["player_1"], 
                                                    last_mem_state = last_mem_state_player_1,
                                                    obs = obs["player_1"], 
                                                    mem_state = env_state.memory_state_player_1,
                                                    params = params)
        }
        return obs, env_state, reward, done, info


################################## REWARD WRAPPER ##################################
class TransformRewardWrapper(GymnaxWrapper):
    """"
    Changes the reward of the environment
    """
    def __init__(self, env: MemoryWrapper, transform_reward: TransformReward):
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
            "player_0": self.transform_reward.convert(team_id=0, last_obs=last_obs["player_0"], obs = obs["player_0"], params = params, player_stats = info["stats"]["player_0"]),
            "player_1": self.transform_reward.convert(team_id=1, last_obs=last_obs["player_1"], obs = obs["player_1"], params = params, player_stats = info["stats"]["player_1"]),
        }
        return obs, state, transformed_reward, done, info
    
    
################################## ACTION WRAPPER ##################################
class TransformActionWrapper(GymnaxWrapper):
    """"
    Changes the action of the environment
    """
    def __init__(self, env: TransformRewardWrapper, transform_action: TransformAction):
        self.transform_action = transform_action
        super().__init__(env)
    
    def action_space(self, params: Optional[EnvParams] = None):
        return gymnax.environments.spaces.Dict(
            dict(player_0=self.transform_action.action_space, player_1=self.transform_action.action_space)
        )

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
    

################################## OBS WRAPPER ##################################
class TransformObsWrapper(GymnaxWrapper):
    """"
    Changes the observation of the environment
    """
    def __init__(self, env: TransformActionWrapper, transform_obs: TransformObs):
        super().__init__(env)
        self.transform_obs = transform_obs
        self.observation_space = self.transform_obs.observation_space
    
    def reset(self, key: chex.PRNGKey, params: EnvParams):
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


################################## LOG WRAPPER ##################################
from purejaxrl.env.tracker import Tracker, PlayerStats
@struct.dataclass
class LogEnvState:
    env_state: Env_Mem_State
    episode_return: chex.Array
    episode_points: chex.Array
    episode_wins: chex.Array
    episode_timestep: int
    global_timestep: int
    episode_stats_player_0: PlayerStats
    episode_stats_player_1: PlayerStats

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""
    def __init__(self, env: TransformObsWrapper, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    def reset(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, params)
        log_env_state = LogEnvState(
            env_state=env_state,
            episode_return=jnp.array([0.0, 0.0], dtype=jnp.float16),
            episode_points=jnp.array([0, 0]),
            episode_wins=jnp.array([0.0, 0.0]),
            episode_timestep=0,
            global_timestep=0,
            episode_stats_player_0=PlayerStats(),
            episode_stats_player_1=PlayerStats(),
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

        # Update episode returns
        new_episode_return = log_env_state.episode_return + jnp.array(
            [reward["player_0"], reward["player_1"]]
        )

        # Update episode points
        new_episode_points = log_env_state.env_state.env_state.team_points + jnp.array(
            [info["stats"]["player_0"].points_gained, info["stats"]["player_1"].points_gained]
        )

        # Update cumulative statistics for each player
        new_episode_stats_player_0 = PlayerStats(
            **{field: getattr(log_env_state.episode_stats_player_0, field) + getattr(info["stats"]["player_0"], field)
               for field in PlayerStats.__dataclass_fields__}
        )
        new_episode_stats_player_1 = PlayerStats(
            **{field: getattr(log_env_state.episode_stats_player_1, field) + getattr(info["stats"]["player_1"], field)
               for field in PlayerStats.__dataclass_fields__}
        )

        # Update wins based on points
        current_episode_wins = jnp.where(
            new_episode_points[0] > new_episode_points[1],
            jnp.array([1.0, 0.0]),
            jnp.where(
                new_episode_points[0] < new_episode_points[1],
                jnp.array([0.0, 1.0]),
                jnp.array([0.5, 0.5]),
            ),
        )
        new_episode_wins = log_env_state.env_state.env_state.team_wins + current_episode_wins

        # Prepare info dictionary
        if self.replace_info:
            info = {}
        info["episode_return"] = new_episode_return
        info["episode_wins"] = new_episode_wins
        info["episode_stats_player_0"] = new_episode_stats_player_0
        info["episode_stats_player_1"] = new_episode_stats_player_1
        info["episode_winner"] = jnp.where(
            new_episode_wins[0] > new_episode_wins[1],
            jnp.array([1, 0]),
            jnp.where(
                new_episode_wins[0] < new_episode_wins[1],
                jnp.array([0, 1]),
                jnp.array([0.5, 0.5]),
            ),
        )
        info["global_timestep"] = (log_env_state.episode_timestep + 1) * (1 - done)
        info["episode_timestep"] = log_env_state.global_timestep + 1
        info["returned_episode"] = done

        # Create new LogEnvState
        new_log_env_state = LogEnvState(
            env_state=env_state,
            episode_return=new_episode_return * (1 - done),
            episode_points=new_episode_points * (1 - done),
            episode_wins=new_episode_wins * (1 - done),
            episode_timestep=(log_env_state.episode_timestep + 1) * (1 - done),
            global_timestep=log_env_state.global_timestep + 1,
            episode_stats_player_0=PlayerStats(
                **{field: getattr(new_episode_stats_player_0, field) * (1-done)
                for field in PlayerStats.__dataclass_fields__}),
            episode_stats_player_1=PlayerStats(
                **{field: getattr(new_episode_stats_player_1, field) * (1-done)
                for field in PlayerStats.__dataclass_fields__}),
        )
        return obs, new_log_env_state, reward, done, info