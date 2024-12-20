from abc import ABC, abstractmethod
import numpy as np
import torch
from src.luxai_s3.state import EnvObs

from agents.reward_shapers.utils import (
    is_alive_mask,
    relic_reward,
    sapped_reward,
    reveal_reward,
    high_energy_reward,
    death_reward,
)


class RewardShaper(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def convert(
        self, previous_obs: EnvObs, r: float, actions, obs: EnvObs, team_id: int
    ) -> torch.Tensor:
        """
        Convert an observation into a tensor representation.

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
        self, previous_obs: EnvObs, r: float, actions, obs: EnvObs, team_id: int
    ) -> torch.Tensor:
        return len(actions) * torch.tensor(r)


class MixingRewardShaper(RewardShaper):
    def convert(
        self, previous_obs: EnvObs, r: float, actions, obs: EnvObs, team_id: int
    ) -> torch.Tensor:
        total_revard_vector = np.zeros((self.env_cfg.max_units, 1), dtype=np.int32)

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
        return torch.tensor(total_revard_vector)
