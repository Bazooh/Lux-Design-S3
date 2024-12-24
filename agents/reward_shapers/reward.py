from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
from src.luxai_s3.state import EnvObs
from src.luxai_s3.env import PlayerAction

from agents.reward_shapers.utils import (
    is_alive_mask,
    relic_reward,
    sapped_reward,
    reveal_reward,
    high_energy_reward,
    death_reward,
)

Reward = np.ndarray[Literal[16], np.dtype[np.float32]]


class RewardShaper(ABC):
    max_agents = 16

    @abstractmethod
    def convert(
        self,
        previous_obs: EnvObs,
        env_reward: np.ndarray[Literal[1], np.dtype[np.float32]],
        actions: PlayerAction,
        obs: EnvObs,
    ) -> Reward:
        """
        Convert an observation into an np.array representation of the reward.

        Args:
            previous_obs (EnvObs): The previous environment observation.
            obs (EnvObs): The environment observation.
            team_id (int): The team identifier.

        Returns:
            torch.Tensor of shape (num_agents, )
        """
        pass


class DefaultRewardShaper(RewardShaper):
    def convert(
        self,
        previous_obs: EnvObs,
        env_reward: np.ndarray[Literal[1], np.dtype[np.float32]],
        actions: PlayerAction,
        obs: EnvObs,
    ) -> Reward:
        return np.array(env_reward).repeat(self.max_agents)


class GreedyRewardShaper(RewardShaper):
    def convert(
        self,
        previous_obs: EnvObs,
        env_reward: np.ndarray[Literal[1], np.dtype[np.float32]],
        actions: PlayerAction,
        obs: EnvObs,
    ) -> Reward:
        return np.array(obs.reward).repeat(self.max_agents)


class MixingRewardShaper(RewardShaper):
    def convert(
        self,
        previous_obs: EnvObs,
        env_reward: np.ndarray[Literal[1], np.dtype[np.float32]],
        actions: PlayerAction,
        obs: EnvObs,
    ) -> Reward:
        total_revard_vector = np.array(self.max_agents, dtype=np.int32)

        # reward for being close to relic
        total_revard_vector += relic_reward(0, previous_obs, obs)

        # reward for sapping
        total_revard_vector += sapped_reward(0, previous_obs, obs, actions)

        # reward for revealing
        total_revard_vector += reveal_reward(0, previous_obs, obs)

        # reward for high energy
        total_revard_vector += high_energy_reward(0, previous_obs, obs)

        # negative reward for death
        total_revard_vector += death_reward(0, previous_obs, obs)

        # reward for revealed
        total_revard_vector += reveal_reward(0, previous_obs, obs)
        return total_revard_vector


class ExploreRewardShaper(RewardShaper):
    def convert(
        self,
        previous_obs: EnvObs,
        env_reward: np.ndarray[Literal[1], np.dtype[np.float32]],
        actions: PlayerAction,
        obs: EnvObs,
    ) -> Reward:
        return np.array(obs.sensor_mask.sum()).repeat(self.max_agents)
