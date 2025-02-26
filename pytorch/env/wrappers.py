import gymnasium as gym
from flax import struct
from typing import Optional, Tuple, Union, Any, Literal
from luxai_s3.wrappers import LuxAIS3GymEnv
from luxai_s3.state import EnvObs, EnvState, EnvParams
from pytorch.env.transform_obs import TransformObs
from pytorch.env.transform_action import TransformAction
from flax import struct
from typing import Any
from pytorch.env.memory import Memory
import numpy as np
PlayerAction = Any

class GymWrapper(object):
    def __init__(self, env):
        self.env = env
        self.num_agents = 2
        self.agents = ["player_0", "player_1"]
    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self.env, name)

################################## SIMPLIFY TRUNCATION WRAPPER ##################################
class SimplifyTruncationWrapper(GymWrapper):
    """"
    Wraps the env from the format:
        (obs, state, reward, terminated_dict, truncated_dict, info)
        to
        (obs, env_state, reward, done, info)
    """

    def __init__(self, env: LuxAIS3GymEnv):
        super().__init__(env)
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, params = self.env.reset(seed=seed, options=options)
        self.env_params = params["full_params"]
        return obs, params
    
    def step(
        self, action: Any
    ) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        obs, reward, terminated_dict, truncated_dict, info = self.env.step(action)
        done = truncated_dict["player_0"] | terminated_dict["player_0"]
        return obs, reward, done, info 


################################## POINTS MAP WRAPPER ##################################
@struct.dataclass
class State_With_Points_Maps:
    env_state: EnvState
    points_map: Any
    def __getattr__(self, name):
        return getattr(self.env_state, name)
    
def compute_points_map(env_state: EnvState) -> Any:
    active_relic_mask = (env_state.relic_spawn_schedule <= env_state.steps) & env_state.relic_nodes_mask
    points_map = (env_state.relic_nodes_map_weights>0) & active_relic_mask[env_state.relic_nodes_map_weights]
    return points_map

class PointsMapWrapper(GymWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_agents = 2
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, params = self.env.reset(seed=seed, options=options)
        self.env_state =  State_With_Points_Maps(self.env.state, compute_points_map(self.env.state))
        return obs, params
    
    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        obs, reward, done, info  = self.env.step(action)
        self.env_state =  State_With_Points_Maps(self.env.state, compute_points_map(self.env.state))
        return obs, reward, done, info 
    
    
################################## MEMORY WRAPPER ##################################
@struct.dataclass
class Env_Mem_State:
    env_state: EnvState
    memory_state_player_0: Any
    memory_state_player_1: Any
    def __getattr__(self, name):
        return getattr(self.env_state, name)
    
class MemoryWrapper(GymWrapper):
    def __init__(self, env: SimplifyTruncationWrapper, memory: Memory):
        super().__init__(env)
        self.memory = memory

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, params = self.env.reset(seed=seed, options=options)
        memory_state_player_0 = self.memory.reset()
        memory_state_player_1 = self.memory.reset()
        self.env_state = Env_Mem_State(
                env_state=self.env_state, 
                memory_state_player_0 = memory_state_player_0, 
                memory_state_player_1 = memory_state_player_1,        
        )
        return obs, params

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        obs, reward, done, info  = self.env.step(action)
        memory_state_player_0 = self.memory.update(obs = obs['player_0'], team_id=0, memory_state=self.env_state.memory_state_player_0, params = self.env_params)
        memory_state_player_1 = self.memory.update(obs = obs['player_1'], team_id=1, memory_state=self.env_state.memory_state_player_1, params = self.env_params)
        self.env_state = Env_Mem_State(
                env_state=self.env_state.env_state, 
                memory_state_player_0 = memory_state_player_0, 
                memory_state_player_1 = memory_state_player_1        
        )
        return obs, reward, done, info 
    
################################## ACTION WRAPPER ##################################
class TransformActionWrapper(GymWrapper):
    """"
    Changes the action of the environment
    """
    def __init__(self, env: MemoryWrapper, transform_action: TransformAction):
        self.transform_action = transform_action
        super().__init__(env)
    
    def action_space(self):
        return self.transform_action.action_space
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, params = self.env.reset(seed=seed, options=options)
        self.prev_obs = obs
        return obs, params
        
    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        current_obs = self.prev_obs
        transform_action = {
            "player_0": self.transform_action.convert(team_id=0, action=action["player_0"], obs = current_obs["player_0"], params = self.env_params),
            "player_1": self.transform_action.convert(team_id=1, action=action["player_1"], obs = current_obs["player_1"], params = self.env_params),
        }
        obs, reward, done, info  = self.env.step(transform_action)
        self.prev_obs = obs
        return obs, reward, done, info
    

################################## OBS WRAPPER ##################################
class TransformObsWrapper(GymWrapper):
    """"
    Changes the observation of the environment
    """
    def __init__(self, env: TransformActionWrapper, transform_obs: TransformObs):
        super().__init__(env)
        self.transform_obs = transform_obs
        self.observation_space = self.transform_obs.observation_space
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, params = self.env.reset(seed=seed, options=options)
        transformed_obs = {
            "player_0": self.transform_obs.convert(team_id=0, obs = obs["player_0"], params = self.env_params, memory_state=self.env_state.memory_state_player_0),
            "player_1": self.transform_obs.convert(team_id=1, obs = obs["player_1"], params = self.env_params, memory_state=self.env_state.memory_state_player_1),
        }
        return transformed_obs, params

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        obs, reward, done, info  = self.env.step(action)
        transformed_obs = {
            "player_0": self.transform_obs.convert(team_id=0, obs = obs["player_0"], params = self.env_params, memory_state=self.env_state.memory_state_player_0),
            "player_1": self.transform_obs.convert(team_id=1, obs = obs["player_1"], params = self.env_params, memory_state=self.env_state.memory_state_player_1),
        }
        return transformed_obs, reward, done, info


################################## LOG WRAPPER ##################################
@struct.dataclass
class LogEnvState:
    env_state: Env_Mem_State
    episode_return_player_0: np.ndarray
    episode_return_player_1: np.ndarray

    def __getattr__(self, name):
        return getattr(self.env_state, name)
    
class LogWrapper(GymWrapper):
    """Log the episode returns and lengths."""
    def __init__(self, env: TransformObsWrapper, replace_info: bool = True):
        super().__init__(env)
        self.replace_info = replace_info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, params = self.env.reset(seed=seed, options=options)
        log_env_state = LogEnvState(
            env_state=self.env_state,
            # episode_return_player_0=np.zeros(shape = (self.env.num_phases), dtype = np.float32),
            # episode_return_player_1=np.zeros(shape = (self.env.num_phases), dtype = np.float32),
            episode_return_player_0=0,
            episode_return_player_1=0
        )
        return obs, log_env_state

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        obs, reward, done, info  = self.env.step(action)
        env_state = self.env_state
        # Update episode returns
        new_episode_return_player_0 = env_state.episode_return_player_0 + reward["player_0"]
        new_episode_return_player_1 = env_state.episode_return_player_1 + reward["player_1"]

        # Prepare info dictionary
        if self.replace_info:
            info = {}

        # info["episode_return_player_0"] = new_episode_return_player_0
        # info["episode_return_player_1"] = new_episode_return_player_1
        # info["dense_stats_player_0"] = env_state.dense_stats_player_0
        # info["dense_stats_player_1"] = env_state.dense_stats_player_1
        # # info["match_stats_player_0"] = env_state.match_stats_player_0
        # # info["match_stats_player_1"] = env_state.match_stats_player_1
        # info["episode_stats_player_0"] = env_state.episode_stats_player_0
        # info["episode_stats_player_1"] = env_state.episode_stats_player_1
        info["returned_episode"] = done

        # Create new LogEnvState
        new_log_env_state = LogEnvState(
            env_state = self.env_state,
            episode_return_player_0 = new_episode_return_player_0 * (1 - done),
            episode_return_player_1 = new_episode_return_player_1 * (1 - done),
        )
        return obs, new_log_env_state, reward, done, info