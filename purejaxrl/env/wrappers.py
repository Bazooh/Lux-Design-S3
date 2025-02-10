import gymnax.environments.spaces
import jax
import chex
from flax import struct
from typing import Optional, Tuple, Union, Any, Literal
from enum import Enum
import jax.numpy as jnp
import gymnax
from luxai_s3.env import LuxAIS3Env, EnvState, EnvParams
from purejaxrl.env.transform_obs import TransformObs
from purejaxrl.env.transform_action import TransformAction
from luxai_s3.env import EnvObs
import jax.numpy as jnp
from flax import struct
import jax, chex
from functools import partial
from typing import Any
from purejaxrl.env.memory import Memory, RelicPointMemoryState
# for recording
from purejaxrl.env.utils import (
    serialize_metadata, 
    serialize_env_params, 
    json_to_html, 
    get_action_masking_from_obs
)
from luxai_s3.state import serialize_env_actions, serialize_env_states
import os, json
PlayerAction = Any
class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
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
    def __getattr__(self, name):
        return getattr(self.env_state, name)
    
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
                memory_state_player_1 = memory_state_player_1,        
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
        memory_state_player_0 = self.memory.update(obs = obs['player_0'], team_id=0, memory_state=env_mem_state.memory_state_player_0, params = params)
        memory_state_player_1 = self.memory.update(obs = obs['player_1'], team_id=1, memory_state=env_mem_state.memory_state_player_1, params = params)
        env_mem_state = Env_Mem_State(
                env_state=env_state, 
                memory_state_player_0 = memory_state_player_0, 
                memory_state_player_1 = memory_state_player_1        
        )
        return obs, env_mem_state, reward, done, info


################################## TRACKER WRAPPER ##################################
@struct.dataclass
class PlayerStats:
    wins: int = 0
    points_gained: int = 0
    points_discovered: int = 0
    relics_discovered: int = 0
    cells_discovered: int = 0
    deaths: int = 0
    collisions: int = 0
    units_moved: int = 0
    energy_gained: int = 0
    sap_tried: int = 0
    sap_available: float = 0

    def __add__(self, other: "PlayerStats"):
        return PlayerStats(
            **{key: value + other.__dict__[key] for key, value in self.__dict__.items()}
        )
    def __mul__(self, x: chex.Array):
        return PlayerStats(**
            {key: value * x for key, value in self.__dict__.items()}
        )

@struct.dataclass
class State_with_Stats:
    env_state: Env_Mem_State
    dense_stats_player_0: PlayerStats
    dense_stats_player_1: PlayerStats
    episode_stats_player_0: PlayerStats
    episode_stats_player_1: PlayerStats
    def __getattr__(self, name):
        return getattr(self.env_state, name)
    
class TrackerWrapper(GymnaxWrapper):
    def __init__(self, env: SimplifyTruncationWrapper):
        super().__init__(env)
        self.stats_names =  PlayerStats.__dataclass_fields__

    @partial(jax.jit, static_argnums=(0, 1))
    def get_player_statistics(
        self,
        team_id: int,
        action: PlayerAction,
        obs: EnvObs,
        mem_state: RelicPointMemoryState,
        last_obs: EnvObs,
        last_mem_state: RelicPointMemoryState,
        params: EnvParams,
    ) -> PlayerStats:
        
        wins_both_players = obs.team_wins - last_obs.team_wins
        wins = jax.lax.select(
            obs.steps % (params.max_steps_in_match + 1) == 0,  # Check if the game is done
            jax.lax.select(
                jnp.sum(wins_both_players) == 0,  # Check if it's a tie
                0,  # Return 0 for a tie
                2*wins_both_players[team_id] - 1,  # Return result otherwise 
            ),
            0 # Return 0 if the match is not done
        )
        
        points_gained = mem_state.points_gained
        
        relics_discovered_image = jnp.fliplr(jnp.triu(jnp.fliplr((mem_state.relics_found_image == 1) & (last_mem_state.relics_found_image != 1))))
        relics_discovered = jax.numpy.sum(relics_discovered_image.astype(jnp.int32))
        points_discovered_image = jnp.fliplr(jnp.triu(jnp.fliplr((mem_state.points_found_image == 1) & (last_mem_state.points_found_image != 1))))
        points_discovered = jax.numpy.sum(points_discovered_image.astype(jnp.int32))
        cells_discovered_image = jnp.fliplr(jnp.triu(jnp.fliplr((mem_state.relics_found_image != 0) & (last_mem_state.relics_found_image == 0))))
        cells_discovered = jax.numpy.sum(cells_discovered_image.astype(jnp.int32))
        
        cumulated_energy  = jnp.sum(obs.units.energy[team_id])
        last_cumulated_energy = jnp.sum(last_obs.units.energy[team_id])

        energy_gained = jax.lax.select(
            last_obs.steps % (params.max_steps_in_match+1) == 0,
            0,
            jnp.maximum(0, cumulated_energy - last_cumulated_energy)
        )
        
        current_positions = obs.units.position[team_id]
        last_positions = last_obs.units.position[team_id]

        last_alive_units_mask = last_obs.units.energy[team_id] > 0
        alive_units_mask = obs.units.energy[team_id] > 0

        moved_units_mask = jax.numpy.any(current_positions != last_positions, axis=-1)
        no_actions_units_mask = jnp.where((action[:,0] == 0) | (action[:,0] == 5), 1, 0) 
        
        units_moved = jax.lax.select(
            obs.steps % (params.max_steps_in_match + 1) == 0,
            0,
            jnp.sum(moved_units_mask & alive_units_mask),
        )

        collisions = jax.lax.select(
            last_obs.steps % (params.max_steps_in_match + 1) == 0,
            0,
            jnp.sum(~no_actions_units_mask & ~moved_units_mask & alive_units_mask) # units that should move but collided to the env
        )
        
        deaths = jax.lax.select(
            last_obs.steps % (params.max_steps_in_match + 1) == 0,
            0,
            jnp.sum(~alive_units_mask & last_alive_units_mask) # units that were alive but are anymore
        )
        
        sap_tried = jnp.sum((action[:, 0] == 5).astype(jnp.int8))
        action_mask = get_action_masking_from_obs(team_id = team_id, obs = obs, sap_range = params.unit_sap_range)
        sap_available = jnp.sum((action_mask[:, 5]).astype(jnp.int8))

        return PlayerStats(
            wins = wins,
            points_gained=jax.lax.stop_gradient(points_gained),
            energy_gained=jax.lax.stop_gradient(energy_gained),
            relics_discovered=jax.lax.stop_gradient(relics_discovered),
            points_discovered=jax.lax.stop_gradient(points_discovered),
            cells_discovered=jax.lax.stop_gradient(cells_discovered),
            units_moved=jax.lax.stop_gradient(units_moved), 
            collisions=jax.lax.stop_gradient(collisions),
            deaths=jax.lax.stop_gradient(deaths),
            sap_tried=jax.lax.stop_gradient(sap_tried),
            sap_available=jax.lax.stop_gradient(sap_available),            
        )
    def reset(self, key: chex.PRNGKey, params: Optional[EnvParams] = None) -> Tuple[chex.Array, State_with_Stats]:
        obs, env_state = self._env.reset(key, params)
        new_state = State_with_Stats(
            env_state=env_state,
            dense_stats_player_0=PlayerStats(),
            dense_stats_player_1=PlayerStats(),
            episode_stats_player_0=PlayerStats(),
            episode_stats_player_1=PlayerStats(),
        )
        return obs, new_state
    
    def step(
        self,
        key: chex.PRNGKey,
        env_state: State_with_Stats,
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, Env_Mem_State, float, bool, dict]:
        last_obs = self._env.get_obs(env_state.env_state)
        obs, new_env_state, reward, done, info = self._env.step(key, env_state.env_state, action, params)
        
        dense_stats_player_0 = self.get_player_statistics(
            team_id=0, 
            action = action["player_0"],
            last_obs=last_obs["player_0"], 
            last_mem_state = env_state.memory_state_player_1, 
            obs = info["final_observation"]["player_0"], 
            mem_state = new_env_state.memory_state_player_0,
            params = params
        )
        dense_stats_player_1 = self.get_player_statistics(
            team_id=1, 
            action = action["player_1"],
            last_obs=last_obs["player_1"], 
            last_mem_state = env_state.memory_state_player_1,
            obs = info["final_observation"]["player_1"], 
            mem_state = new_env_state.memory_state_player_1,
            params = params
        )
        episode_stats_player_0 = env_state.episode_stats_player_0 + dense_stats_player_0
        episode_stats_player_1 = env_state.episode_stats_player_1 + dense_stats_player_1
        
        new_env_state = State_with_Stats(
            env_state=new_env_state,
            dense_stats_player_0=dense_stats_player_0,
            dense_stats_player_1=dense_stats_player_1,
            episode_stats_player_0=episode_stats_player_0 * (1-done),
            episode_stats_player_1=episode_stats_player_1 * (1-done),
        )
        return obs, new_env_state, reward, done, info


################################## REWARD WRAPPER ##################################
class RewardType(Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    SPARSE_DELTA = "sparse_delta"

class RewardObject:
    def __init__(self, reward_type: RewardType, reward_weights: dict[str, float]):
        if not isinstance(reward_type, RewardType):
            raise ValueError("reward_type must be an instance of RewardType Enum")
        self.reward_type = reward_type
        self.reward_weights = reward_weights
    
    def __repr__(self):
        return f"RewardObject(type={self.reward_type.value}, weights={self.reward_weights})"
    def compute_reward(
        self, 
        done: bool,
        match_end: bool,
        dense_stats_player_0: PlayerStats, 
        dense_stats_player_1: PlayerStats,
        episode_stats_player_0: PlayerStats,
        episode_stats_player_1: PlayerStats,
    ) -> dict[str, float]:
        if self.reward_type == RewardType.DENSE: 
            return {
                "player_0": sum([getattr(dense_stats_player_0, stat) * weight for stat, weight in self.reward_weights.items()]),
                "player_1": sum([getattr(dense_stats_player_1, stat) * weight for stat, weight in self.reward_weights.items()]),
            }
        elif self.reward_type == RewardType.SPARSE:
            return {
                "player_0": sum([getattr(episode_stats_player_0, stat) * weight for stat, weight in self.reward_weights.items()]) if done else 0,
                "player_1": sum([getattr(episode_stats_player_1, stat) * weight for stat, weight in self.reward_weights.items()]) if done else 0,
            }
        elif self.reward_type == RewardType.SPARSE_DELTA:
            reward =  {
                "player_0": sum([getattr(episode_stats_player_0, stat) * weight for stat, weight in self.reward_weights.items()]) if done else 0,
                "player_1": sum([getattr(episode_stats_player_1, stat) * weight for stat, weight in self.reward_weights.items()]) if done else 0,
            }
            return {
                "player_0": jnp.sqrt(reward["player_0"] - reward["player_1"]),
                "player_1": jnp.sqrt(reward["player_0"] - reward["player_1"]),
            }
class TransformRewardWrapper(GymnaxWrapper):
    """"
    Changes the reward of the environment
    """
    def __init__(self, env: TrackerWrapper, reward_phases: list[RewardObject]):
        super().__init__(env)
        self.reward_phases = reward_phases

    def step(
        self,
        key: chex.PRNGKey,
        env_mem_state: Env_Mem_State,
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, Env_Mem_State, float, bool, dict]:
        obs, state, _, done, info = self._env.step(key, env_mem_state, action, params)
        match_end = state.steps % (params.max_steps_in_match + 1) == 0
        transformed_reward = [
            self.reward_phases[i].compute_reward(
                done = done,
                match_end = match_end,
                dense_stats_player_0 = state.dense_stats_player_0,
                dense_stats_player_1 = state.dense_stats_player_1,
                episode_stats_player_0 = state.episode_stats_player_0,
                episode_stats_player_1 = state.episode_stats_player_1,
            ) for i in range(len(self.reward_phases))
        ]
        transformed_reward = {
            "player_0": jnp.array([reward["player_0"] for reward in transformed_reward]),
            "player_1": jnp.array([reward["player_1"] for reward in transformed_reward]),
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
        transform_action = {
            "player_0": self.transform_action.convert(team_id=0, action=action["player_0"], obs = current_obs["player_0"], params = params),
            "player_1": self.transform_action.convert(team_id=1, action=action["player_1"], obs = current_obs["player_1"], params = params),
        }
        obs, env_state, reward, done, info = self._env.step(
            key, env_mem_state, transform_action, params
        )
        return obs, env_state, reward, done, info 
    

################################## OBS WRAPPER ##################################
class TransformObsWrapper(GymnaxWrapper):
    """"
    Changes the observation of the environment
    """
    def __init__(self, env: TransformActionWrapper, transform_obs: TransformObs):
        super().__init__(env)
        self._env = env
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
@struct.dataclass
class LogEnvState:
    env_state: Env_Mem_State
    episode_return: chex.Array
    episode_stats_player_0: PlayerStats
    episode_stats_player_1: PlayerStats

    def __getattr__(self, name):
        return getattr(self.env_state, name)
    
class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""
    def __init__(self, env: TransformObsWrapper, replace_info: bool = True):
        super().__init__(env)
        self.replace_info = replace_info

    def reset(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, params)
        log_env_state = LogEnvState(
            env_state=env_state,
            episode_return=jnp.array([0.0, 0.0], dtype=jnp.float32),
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
        new_episode_return = log_env_state.episode_return + jnp.array([reward["player_0"][0], reward["player_1"][0]])

        # Prepare info dictionary
        if self.replace_info:
            info = {}
        info["episode_return"] = new_episode_return
        info["episode_stats_player_0"] = log_env_state.episode_stats_player_0
        info["episode_stats_player_1"] = log_env_state.episode_stats_player_1
        info["returned_episode"] = done

        # Create new LogEnvState
        new_log_env_state = LogEnvState(
            env_state = env_state,
            episode_return = new_episode_return * (1 - done),
            episode_stats_player_0 = log_env_state.episode_stats_player_0 * (1-done),
            episode_stats_player_1 = log_env_state.episode_stats_player_1 * (1-done),
        )
        return obs, new_log_env_state, reward, done, info