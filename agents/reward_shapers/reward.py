from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
import torch
from src.luxai_s3.env import PlayerAction

from agents.obs import Obs

# from agents.reward_shapers.utils import (
#     is_alive_mask,
#     relic_reward,
#     sapped_reward,
#     reveal_reward,
#     high_energy_reward,
#     death_reward,
# )

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
        pass


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
        return np.array(reward)


class MixingRewardShaper(RewardShaper):
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
        total_revard_vector = np.array(self.max_agents, dtype=np.float32)

        # reward for being close to relic
        # total_revard_vector += relic_reward(0, obs, next_obs)

        # reward for sapping
        # total_revard_vector += sapped_reward(0, obs, next_obs, actions)

        # reward for revealing
        # total_revard_vector += reveal_reward(0, obs, next_obs)

        # reward for high energy
        # total_revard_vector += high_energy_reward(0, obs, next_obs)

        # negative reward for death
        # total_revard_vector += death_reward(0, obs, next_obs)

        # reward for revealed
        # total_revard_vector += reveal_reward(0, obs, next_obs)
        return total_revard_vector


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
