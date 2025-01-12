from dataclasses import dataclass

import torch

from agents.reward_shapers.reward import RewardShaper
from agents.rl_agent import symetric_action
from agents.tensor_converters.tensor import TensorConverter
from luxai_runner.episode import json_to_html
from luxai_s3.state import serialize_env_actions, serialize_env_states
from luxai_s3.wrappers import LuxAIS3GymEnv, PlayerName
from agents.obs import EnvParams, GodObs, VecEnvParams
from typing import Any, Callable, Literal
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import os
import flax
import flax.serialization
from pathlib import Path
import numpy as np
from agents.base_agent import N_Actions, N_Agents, N_Players

BatchSize = int
N_Channels = int

PlayerAgentMask = np.ndarray[tuple[N_Players, N_Agents], np.dtype[np.bool_]]

VecPlayerAction = np.ndarray[
    tuple[BatchSize, N_Players, N_Agents, N_Actions], np.dtype[np.int32]
]
VecPlayerObservation = np.ndarray[
    tuple[BatchSize, N_Players, N_Channels, 24, 24], np.dtype[np.float32]
]
VecPlayerReward = np.ndarray[
    tuple[BatchSize, N_Players, N_Agents], np.dtype[np.float32]
]
VecPlayerAgentMask = np.ndarray[
    tuple[BatchSize, N_Players, N_Agents], np.dtype[np.bool_]
]


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


def observation_space_from_player_observation_space(
    player_observation_space: gym.spaces.Space,
) -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "player_0": player_observation_space,
            "player_1": player_observation_space,
        }
    )


class EnvInterfaceForVec(LuxAIS3GymEnv):
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

    def __init__(
        self,
        tensor_converter_instantiator: Callable[[], TensorConverter],
        reward_shaper: RewardShaper,
    ):
        self.tensor_converter_0 = tensor_converter_instantiator()
        self.tensor_converter_1 = tensor_converter_instantiator()
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(2, self.tensor_converter_0.n_channels(), 24, 24),
            dtype=np.float32,
        )

        self.reward_shaper = reward_shaper
        super().__init__(numpy_output=True)

    def to_tensor(self, obs: GodObs) -> np.ndarray:
        return np.stack(
            (
                self.tensor_converter_0.convert(obs.player_0, 0),
                self.tensor_converter_1.convert(obs.player_1, 1),
            )
        )

    def reset(  # type: ignore
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, EnvParams]:
        self.tensor_converter_0.reset_memory()
        self.tensor_converter_1.reset_memory()

        obs, info = super().reset(seed=seed, options=options)
        self.obs = GodObs.from_dict(obs)
        self.env_param = EnvParams.from_dict(info["params"])

        self.reward = np.zeros((2, 16), dtype=np.float32)

        return self.to_tensor(self.obs), info["params"]

    def step(  # type: ignore
        self, action: np.ndarray
    ) -> tuple[
        np.ndarray,
        float,
        float,
        float,
        dict[str, Any],
    ]:
        awake = np.stack(
            (self.obs.player_0.units_mask[0], self.obs.player_1.units_mask[1])
        )

        obs, reward, terminated, truncated, info = super().step(
            {"player_0": action[0], "player_1": symetric_action(action[1])}
        )
        self.obs = GodObs.from_dict(obs)
        tensor_obs = self.to_tensor(self.obs)

        self.tensor_converter_0.update_memory(self.obs.player_0, 0)
        self.tensor_converter_1.update_memory(self.obs.player_1, 1)

        done = np.stack(
            (self.obs.player_0.units_mask[0], self.obs.player_1.units_mask[1])
        )

        self.reward = np.stack(
            (
                self.reward_shaper.convert(
                    self.env_param,
                    self.reward[0],
                    action[0],
                    self.obs.player_0,
                    tensor_obs[0],
                    0,
                ),
                self.reward_shaper.convert(
                    self.env_param,
                    self.reward[1],
                    action[1],
                    self.obs.player_1,
                    tensor_obs[1],
                    1,
                ),
            )
        )
        info["reward"] = self.reward
        info["awake"] = awake
        info["done"] = done
        info["game_finished"] = truncated["player_0"].item()

        return tensor_obs, 0, 0, 0, info


# TODO : Implement this class
class VecEnvInterface(SyncVectorEnv):
    def __init__(
        self,
        n_envs: int,
        tensor_converter_instantiator: Callable[[], TensorConverter],
        reward_shaper: RewardShaper,
    ):
        self.n_envs = n_envs
        super().__init__(
            [
                lambda: EnvInterfaceForVec(tensor_converter_instantiator, reward_shaper)
                for _ in range(n_envs)
            ]  # type: ignore
        )

    def reset(  # type: ignore
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VecPlayerObservation, VecEnvParams]:
        obs, config = super().reset(seed=seed, options=options)
        return obs, VecEnvParams.from_dict(config)

    def step(  # type: ignore
        self, actions: VecPlayerAction
    ) -> tuple[
        torch.Tensor,
        VecPlayerReward,
        VecPlayerAgentMask,
        VecPlayerAgentMask,
        dict[str, Any],
    ]:
        obs, _, _, _, info = super().step(actions)
        return (
            torch.from_numpy(obs),
            np.stack(info["reward"]),
            np.stack(info["awake"]),
            np.stack(info["done"]),
            info,
        )
