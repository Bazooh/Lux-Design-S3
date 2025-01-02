from dataclasses import dataclass
from luxai_runner.episode import json_to_html
from luxai_s3.state import serialize_env_actions, serialize_env_states
from luxai_s3.wrappers import LuxAIS3GymEnv, PlayerName
from agents.obs import EnvParams, GodObs
from typing import Any, Literal
import gymnasium as gym
from gym.vector import SyncVectorEnv
import os
import flax
import flax.serialization
from pathlib import Path
import numpy as np


class EnvInterface(LuxAIS3GymEnv):
    def __init__(self):
        super().__init__(numpy_output=True)

    def reset(  # type: ignore
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[GodObs, EnvParams]:
        obs, config = super().reset(seed=seed, options=options)
        return GodObs.from_dict(obs), EnvParams.from_dict(config["params"])

    def step(  # type: ignore
        self, actions
    ) -> tuple[
        GodObs,
        dict[PlayerName, np.ndarray[Literal[1], np.dtype[np.float32]]],
        dict[PlayerName, np.ndarray[Literal[1], np.dtype[np.bool_]]],
        dict[PlayerName, np.ndarray[Literal[1], np.dtype[np.bool_]]],
        dict[str, Any],
    ]:
        obs, reward, terminated, truncated, info = super().step(actions)
        return GodObs.from_dict(obs), reward, terminated, truncated, info


@dataclass
class Episode:
    metadata: dict[str, Any]
    states: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    params: dict[str, Any]

    def __init__(self):
        self.metadata = {}
        self.states = []
        self.actions = []
        self.params = {}


class RecordEpisode(gym.Wrapper):
    def __init__(
        self,
        save_dir: str,
        save_on_close: bool = True,
        save_on_reset: bool = True,
    ):
        self.env = LuxAIS3GymEnv(numpy_output=True)
        super().__init__(self.env)
        self.episode = Episode()
        self.episode_id = 0
        self.save_dir = save_dir
        self.save_on_close = save_on_close
        self.save_on_reset = save_on_reset
        self.episode_steps = 0
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def reset(  # type: ignore
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[GodObs, EnvParams]:
        if self.save_on_reset and self.episode_steps > 0:
            self._save_episode_and_reset()
        obs, config = self.env.reset(seed=seed, options=options)

        self.episode.metadata["seed"] = seed
        self.episode.params = flax.serialization.to_state_dict(config["full_params"])
        self.episode.states.append(config["state"])
        return GodObs.from_dict(obs), EnvParams.from_dict(config["params"])

    def step(  # type: ignore
        self, action: Any
    ) -> tuple[
        GodObs,
        dict[PlayerName, np.ndarray[Literal[1], np.dtype[np.float32]]],
        dict[PlayerName, np.ndarray[Literal[1], np.dtype[np.bool_]]],
        dict[PlayerName, np.ndarray[Literal[1], np.dtype[np.bool_]]],
        dict[str, Any],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        self.episode.states.append(info["final_state"])
        self.episode.actions.append(action)
        return GodObs.from_dict(obs), reward, terminated, truncated, info

    def serialize_episode_data(self, episode: Episode | None = None):
        if episode is None:
            episode = self.episode
        ret = dict()
        ret["observations"] = serialize_env_states(episode.states)  # type: ignore
        if len(episode.actions) > 0:
            ret["actions"] = serialize_env_actions(episode.actions)
        ret["metadata"] = episode.metadata
        ret["params"] = episode.params
        return ret

    def save_episode(self, save_path: str):
        episode = self.serialize_episode_data()
        with open(save_path, "w") as f:
            f.write(json_to_html(episode))
        self.episode = Episode()

    def _save_episode_and_reset(self):
        """saves to generated path based on self.save_dir and episoe id and updates relevant counters"""
        self.save_episode(
            os.path.join(self.save_dir, f"episode_{self.episode_id}.html")
        )
        self.episode_id += 1
        self.episode_steps = 0

    def close(self):
        if self.save_on_close and self.episode_steps > 0:
            self._save_episode_and_reset()


class EnvInterfaceForVec(LuxAIS3GymEnv):
    player_observation_space = gym.spaces.Dict(
        {
            "units": gym.spaces.Dict(
                {
                    "position": gym.spaces.Box(
                        low=-1, high=23, shape=(2, 16, 2), dtype=np.int32
                    ),
                    "energy": gym.spaces.Box(
                        low=-1, high=400, shape=(2, 16), dtype=np.int32
                    ),
                }
            ),
            "units_mask": gym.spaces.MultiBinary((2, 16)),
            "sensor_mask": gym.spaces.MultiBinary((24, 24)),
            "map_features": gym.spaces.Dict(
                {
                    "energy": gym.spaces.Box(
                        low=-20, high=20, shape=(24, 24), dtype=np.int32
                    ),
                    "tile_type": gym.spaces.Box(
                        low=-1, high=2, shape=(24, 24), dtype=np.int32
                    ),
                }
            ),
            "relic_nodes": gym.spaces.Box(
                low=-1, high=23, shape=(6, 2), dtype=np.int32
            ),
            "relic_nodes_mask": gym.spaces.MultiBinary((6,)),
            "team_points": gym.spaces.Box(low=0, high=1000, shape=(2,), dtype=np.int32),
            "team_wins": gym.spaces.Box(low=0, high=5, shape=(2,), dtype=np.int32),
            "steps": gym.spaces.Discrete(500),
            "match_steps": gym.spaces.Discrete(100),
        }
    )
    observation_space = gym.spaces.Dict(
        {
            "player_0": player_observation_space,
            "player_1": player_observation_space,
        }
    )

    player_action_space = gym.spaces.Box(
        low=np.array([[0, -4, -4]] * 16),
        high=np.array([[4, 4, 4]] * 16),
        shape=(16, 3),
        dtype=np.int32,
    )
    action_space = gym.spaces.Box(
        low=np.array([[[0, -4, -4]] * 16] * 2),
        high=np.array([[[4, 4, 4]] * 16] * 2),
        shape=(2, 16, 3),
        dtype=np.int32,
    )

    def __init__(self):
        super().__init__(numpy_output=True)

    def step(self, action: np.ndarray):  # type: ignore
        obs, reward, terminated, truncated, info = super().step(
            {"player_0": action[0], "player_1": action[1]}
        )
        return obs, 0, terminated, truncated, info


# TODO : Implement this class
class EnvVectorInterface:
    def __init__(self, n_envs: int):
        self.envs = SyncVectorEnv([lambda: EnvInterfaceForVec() for _ in range(n_envs)])  # type: ignore

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> list[tuple[GodObs, dict[str, Any]]]: ...
