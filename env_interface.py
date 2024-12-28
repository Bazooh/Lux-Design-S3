from luxai_s3.wrappers import LuxAIS3GymEnv, PlayerName
from agents.obs import GodObs
from typing import Any, Literal

import numpy as np


class EnvInterface(LuxAIS3GymEnv):
    def __init__(self):
        super().__init__(numpy_output=True)

    def reset(self) -> tuple[GodObs, dict[str, Any]]:  # type: ignore
        obs, config = super().reset()
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
