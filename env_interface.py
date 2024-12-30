from luxai_s3.wrappers import LuxAIS3GymEnv, PlayerName
from agents.obs import GodObs
from typing import Any, Literal
import gymnasium as gym
from gym.vector import SyncVectorEnv

import numpy as np


class EnvInterface(LuxAIS3GymEnv):
    def __init__(self):
        super().__init__(numpy_output=True)

    def reset(  # type: ignore
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[GodObs, dict[str, Any]]:
        obs, config = super().reset(seed=seed, options=options)
        return GodObs.from_dict(obs), config

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
