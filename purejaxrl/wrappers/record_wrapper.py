from functools import partial
from typing import Optional, Tuple, Union, Any, Literal
import jax, chex
from purejaxrl.wrappers.base_wrappers import GymnaxWrapper
import os, json
from luxai_runner.episode import json_to_html
from luxai_s3.env import LuxAIS3Env, EnvState, EnvParams, PlayerAction
from luxai_s3.state import serialize_env_actions, serialize_env_states
from luxai_s3.params import serialize_env_params

def serialize_metadata(metadata: dict) -> dict:
    serialized = {}
    for key, value in metadata.items():
        if isinstance(value, (int, float, str, bool)):  # Directly serializable types
            serialized[key] = value
        elif isinstance(value, jax.Array) or isinstance(value, jax.numpy.ndarray):  # JAX or NumPy arrays
            serialized[key] = jax.device_get(value).tolist()
        elif isinstance(value, dict):  # Nested dictionaries
            serialized[key] = serialize_metadata(value)
        elif isinstance(value, (list, tuple)):  # Lists or tuples
            serialized[key] = [serialize_metadata(v) if isinstance(v, dict) else v for v in value]
    return serialized

class RecordEpisode(GymnaxWrapper): # adapted from the gym record wrapper
    def __init__(
        self,
        env: Any,
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
        self.episode["states"].append(env_state.env_state)
        return obs, env_state

    def step(
        self,
        key: chex.PRNGKey,
        env_state: EnvState,
        action: PlayerAction,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, env_state, action, params
        )
        self.episode_steps += 1
        self.episode["states"].append(info["final_state"])
        self.episode["actions"].append(action)
        return obs, env_state, reward, done, info
    
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
