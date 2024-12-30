from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
import torch
from src.luxai_s3.env import PlayerAction

from agents.obs import Obs

Reward = np.ndarray[Literal[16], np.dtype[np.float32]]


class RewardShaper(ABC):
    max_agents = 16

    @abstractmethod
    def convert(
        self,
        obs: Obs,
        tensor_obs: torch.Tensor,
        env_reward: float,
        actions: PlayerAction,
        next_obs: Obs,
        next_tensor_obs: torch.Tensor,
        team_id: int,
    ) -> Reward:
        """
        Convert an observation into an np.array representation of the reward.

        Args:
            previous_obs (Obs): The previous environment observation.
            env_reward (float): The environment reward.
            obs (Obs): The environment observation.
            team_id (int): The team identifier.

        Returns:
            torch.Tensor of shape (num_agents, )
        """

        def __add__(self, other: "RewardShaper") -> "RewardShaper":
            return RewardShaperAdder(self, other)

        def __mul__(self, scalar: float) -> "RewardShaper":
            return RewardShaperScaler(self, scalar)

        def __rmul__(self, scalar: float) -> "RewardShaper":
            return RewardShaperScaler(self, scalar)


class RewardShaperAdder(RewardShaper):
    def __init__(self, a: RewardShaper, b: RewardShaper):
        self.a = a
        self.b = b

    def convert(self, *args, **kwargs) -> Reward:
        return self.a.convert(*args, **kwargs) + self.b.convert(*args, **kwargs)


class RewardShaperScaler(RewardShaper):
    def __init__(self, a: RewardShaper, scalar: float):
        self.a = a
        self.scalar = scalar

    def convert(self, *args, **kwargs) -> Reward:
        return self.a.convert(*args, **kwargs) * self.scalar


class DefaultRewardShaper(RewardShaper):
    def convert(
        self,
        obs: Obs,
        tensor_obs: torch.Tensor,
        env_reward: float,
        actions: PlayerAction,
        next_obs: Obs,
        next_tensor_obs: torch.Tensor,
        team_id: int,
    ) -> Reward:
        return np.array(env_reward).repeat(self.max_agents)


class GreedyRewardShaper(RewardShaper):
    def convert(
        self,
        obs: Obs,
        tensor_obs: torch.Tensor,
        env_reward: float,
        actions: PlayerAction,
        next_obs: Obs,
        next_tensor_obs: torch.Tensor,
        team_id: int,
    ) -> Reward:
        point_tensor = next_tensor_obs[4].relu()
        n_unit_tensor = (next_tensor_obs[7:23] > 0).sum(dim=0)
        units_position_tensor = torch.from_numpy(next_obs.units.position[team_id]).int()

        n_units = n_unit_tensor[
            units_position_tensor[:, 0], units_position_tensor[:, 1]
        ]

        reward = (
            point_tensor[units_position_tensor[:, 0], units_position_tensor[:, 1]]
            / n_units
        )
        reward[n_units == 0] = -1
        return reward.numpy()


class ExploreRewardShaper(RewardShaper):
    def convert(
        self,
        obs: Obs,
        tensor_obs: torch.Tensor,
        env_reward: float,
        actions: PlayerAction,
        next_obs: Obs,
        next_tensor_obs: torch.Tensor,
        team_id: int,
    ) -> Reward:
        return np.array(next_obs.sensor_mask.sum()).repeat(self.max_agents)


class GreedyExploreRewardShaper(RewardShaper):
    def convert(
        self,
        obs: Obs,
        tensor_obs: torch.Tensor,
        env_reward: float,
        actions: PlayerAction,
        next_obs: Obs,
        next_tensor_obs: torch.Tensor,
        team_id: int,
    ) -> Reward: ...
