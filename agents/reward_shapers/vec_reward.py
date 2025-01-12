from abc import ABC, abstractmethod
from typing import Literal, Tuple, TYPE_CHECKING
import numpy as np
from agents.obs import VecObs, VecEnvParams
import torch

if TYPE_CHECKING:
    from env_interface import VecPlayerAction


VecReward = np.ndarray[Tuple[int, Literal[16]], np.dtype[np.float32]]


class VecRewardShaper(ABC):
    max_agents = 16

    def __init__(self, n_envs: int):
        self.n_envs = n_envs

    @abstractmethod
    def convert(
        self,
        env_params: VecEnvParams,
        last_reward: VecReward,
        obs: VecObs,
        tensor_obs: torch.Tensor,
        actions: "VecPlayerAction",
        next_obs: VecObs,
        next_tensor_obs: torch.Tensor,
        teams_id: np.ndarray,
    ) -> VecReward:
        """Convert an observation into an np.array representation of the reward."""

    def __add__(self, other: "VecRewardShaper") -> "VecRewardShaper":
        return VecRewardShaperAdder(self, other)

    def __mul__(self, scalar: float) -> "VecRewardShaper":
        return VecRewardShaperScaler(self, scalar)

    def __rmul__(self, scalar: float) -> "VecRewardShaper":
        return VecRewardShaperScaler(self, scalar)


class VecRewardShaperAdder(VecRewardShaper):
    def __init__(self, a: VecRewardShaper, b: VecRewardShaper):
        self.a = a
        self.b = b

    def convert(self, *args, **kwargs) -> VecReward:
        return self.a.convert(*args, **kwargs) + self.b.convert(*args, **kwargs)


class VecRewardShaperScaler(VecRewardShaper):
    def __init__(self, a: VecRewardShaper, scalar: float):
        self.a = a
        self.scalar = scalar

    def convert(self, *args, **kwargs) -> VecReward:
        return self.a.convert(*args, **kwargs) * self.scalar


class VecGreedyRewardShaper(VecRewardShaper):
    def convert(
        self,
        env_params: VecEnvParams,
        last_reward: VecReward,
        obs: VecObs,
        tensor_obs: torch.Tensor,
        actions: "VecPlayerAction",
        next_obs: VecObs,
        next_tensor_obs: torch.Tensor,
        teams_id: np.ndarray,
    ) -> VecReward:
        point_tensor = next_tensor_obs[:, 4].relu()
        n_unit_tensor = (next_tensor_obs[:, 7:23] > 0).sum(dim=1)
        units_position_tensor = torch.from_numpy(
            next_obs.units.position[teams_id]
        ).int()

        n_units = n_unit_tensor[
            :, units_position_tensor[:, :, 0], units_position_tensor[:, :, 1]
        ]

        reward = (
            point_tensor[
                :, units_position_tensor[:, :, 0], units_position_tensor[:, :, 1]
            ]
            / n_units
        )
        reward[n_units == 0] = -1
        return reward.numpy()


class VecDistanceToNearestRelicRewardShaper(VecRewardShaper):
    def convert(
        self,
        env_params: VecEnvParams,
        last_reward: VecReward,
        obs: VecObs,
        tensor_obs: torch.Tensor,
        actions: "VecPlayerAction",
        next_obs: VecObs,
        next_tensor_obs: torch.Tensor,
        teams_id: np.ndarray,
    ) -> VecReward:
        reward = np.zeros((self.n_envs, self.max_agents), dtype=np.float32)
        mask = next_obs.relic_nodes_mask.any(1)

        reward[~mask] = -np.ones(self.max_agents, dtype=np.float32)

        units_position_tensor = torch.from_numpy(
            next_obs.units.position[teams_id]
        ).int()
        relic_position_tensor = torch.from_numpy(next_obs.relic_nodes).int()

        x_diff = (
            units_position_tensor[mask, :, 0].unsqueeze(2)
            - relic_position_tensor[mask, :, 0]
        )
        y_diff = (
            units_position_tensor[mask, :, 1].unsqueeze(2)
            - relic_position_tensor[mask, :, 1]
        )

        distance = ((x_diff**2 + y_diff**2) / (2 * 24**2)).sqrt().min(dim=2).values
        clamped_distance = distance.clamp(min=(1 / 12) ** 2)

        reward[mask] = (1 - clamped_distance).numpy()

        return reward
