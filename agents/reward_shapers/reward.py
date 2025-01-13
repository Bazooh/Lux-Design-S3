from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
import torch
from src.luxai_s3.env import PlayerAction

from agents.obs import EnvParams, Obs

Reward = np.ndarray[Literal[16], np.dtype[np.float32]]


def n_agents_alive(obs: Obs, team_id: int) -> int:
    return obs.units_mask[team_id].sum()


class RewardShaper(ABC):
    max_agents = 16

    @abstractmethod
    def convert(
        self,
        env_params: EnvParams,
        last_reward: Reward,
        actions: PlayerAction,
        next_obs: Obs,
        next_tensor_obs: np.ndarray,
        team_id: int,
    ) -> Reward:
        """Convert an observation into an np.array representation of the reward."""

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
        env_params: EnvParams,
        last_reward: Reward,
        actions: PlayerAction,
        next_obs: Obs,
        next_tensor_obs: np.ndarray,
        team_id: int,
    ) -> Reward:
        return next_obs.team_points[team_id].repeat(self.max_agents) - last_reward


class GreedyRewardShaper(RewardShaper):
    def convert(
        self,
        env_params: EnvParams,
        last_reward: Reward,
        actions: PlayerAction,
        next_obs: Obs,
        next_tensor_obs: np.ndarray,
        team_id: int,
    ) -> Reward:
        point_tensor = next_tensor_obs[-1] * (next_tensor_obs[-1] > 0)
        n_unit_tensor = (next_tensor_obs[-18:-2] > 0).sum(0)

        n_units = n_unit_tensor[
            next_obs.units.position[team_id, :, 0],
            next_obs.units.position[team_id, :, 1],
        ]
        n_units[n_units == 0] = 1

        reward = (
            point_tensor[
                next_obs.units.position[team_id, :, 0],
                next_obs.units.position[team_id, :, 1],
            ]
            / n_units
        )
        return reward


class DistanceToNearestRelicRewardShaper(RewardShaper):
    def convert(
        self,
        env_params: EnvParams,
        last_reward: Reward,
        actions: PlayerAction,
        next_obs: Obs,
        next_tensor_obs: np.ndarray,
        team_id: int,
    ) -> Reward:
        if not next_obs.relic_nodes_mask.any():
            return np.zeros(self.max_agents, dtype=np.float32)

        units_position_tensor = torch.from_numpy(next_obs.units.position[team_id]).int()
        relic_position_tensor = torch.from_numpy(
            next_obs.relic_nodes[next_obs.relic_nodes_mask]
        ).int()

        x_diff = units_position_tensor[:, 0].unsqueeze(1) - relic_position_tensor[:, 0]
        y_diff = units_position_tensor[:, 1].unsqueeze(1) - relic_position_tensor[:, 1]

        distance = ((x_diff**2 + y_diff**2) / (2 * 24**2)).sqrt().min(dim=1).values
        clamped_distance = distance.clamp(min=(1 / 12) ** 2)

        return (1 - clamped_distance).numpy()


class ExploreRewardShaper(RewardShaper):
    def convert(
        self,
        env_params: EnvParams,
        last_reward: Reward,
        actions: PlayerAction,
        next_obs: Obs,
        next_tensor_obs: np.ndarray,
        team_id: int,
    ) -> Reward:
        return (next_obs.sensor_mask.sum() / n_agents_alive(next_obs, team_id)).repeat(
            self.max_agents
        ) - last_reward


class GreedyExploreRewardShaper(RewardShaper):
    def convert(
        self,
        env_params: EnvParams,
        last_reward: Reward,
        actions: PlayerAction,
        next_obs: Obs,
        next_tensor_obs: np.ndarray,
        team_id: int,
    ) -> Reward:
        vision = env_params.unit_sensor_range
        x_min = np.stack(
            (
                np.zeros(16, dtype=np.int32),
                next_obs.units.position[team_id, :, 0] - vision,
            )
        ).max(0)
        x_max = np.stack(
            (
                np.ones(16, dtype=np.int32) * env_params.map_width,
                next_obs.units.position[team_id, :, 0] + vision + 1,
            )
        ).min(0)
        y_min = np.stack(
            (
                np.zeros(16, dtype=np.int32),
                next_obs.units.position[team_id, :, 1] - vision,
            )
        ).max(0)
        y_max = np.stack(
            (
                np.ones(16, dtype=np.int32) * env_params.map_height,
                next_obs.units.position[team_id, :, 1] + vision + 1,
            )
        ).min(0)

        agents_sensor = np.zeros(
            (16, env_params.map_height, env_params.map_width), dtype=np.int32
        )
        for i in range(16):
            if next_obs.units_mask[team_id, i]:
                agents_sensor[i, x_min[i] : x_max[i], y_min[i] : y_max[i]] = (
                    next_obs.sensor_mask[x_min[i] : x_max[i], y_min[i] : y_max[i]]
                )

        max_vision = (2 * vision + 1) ** 2

        agents_sensor_sum = agents_sensor.sum(0)
        agents_sensor_sum[agents_sensor_sum == 0] = 1
        reward = np.zeros(16, dtype=np.float32)
        for i in range(16):
            if next_obs.units_mask[team_id, i]:
                reward[i] = (agents_sensor[i] / agents_sensor_sum).sum() / max_vision

        return reward
