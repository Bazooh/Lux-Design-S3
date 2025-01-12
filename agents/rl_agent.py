from luxai_s3.env import PlayerAction
from agents.tensor_converters.tensor import TensorConverter, BasicTensorConverter
from agents.reward_shapers.reward import (
    # GreedyExploreRewardShaper,
    DistanceToNearestRelicRewardShaper,
    GreedyRewardShaper,
    RewardShaper,
)
from agents.base_agent import Agent, N_Actions, N_Agents
from agents.obs import EnvParams, Obs, VecObs

from abc import ABC, abstractmethod
import random
import numpy as np
import torch
import torch.nn as nn


def symetric_action_not_vectorized(action: int) -> int:
    return [0, 3, 4, 1, 2][action]


symetric_action = np.vectorize(symetric_action_not_vectorized)


class RLAgent(Agent, ABC):
    def __init__(
        self,
        player: str,
        env_params: EnvParams,
        device: str,
        model: torch.nn.Module,
        tensor_converter: TensorConverter,
        reward_shaper: RewardShaper,
        symetric_player_1: bool = True,
    ) -> None:
        super().__init__(player, env_params)
        self.device = device
        self.model = model
        self.tensor_converter = tensor_converter
        self.reward_shaper = reward_shaper
        self.symetric_player_1 = symetric_player_1

    def update_memory(self, obs: Obs) -> None:
        self.tensor_converter.update_memory(
            VecObs.from_obs(obs), np.array([self.team_id])
        )

    def actions(
        self, obs: Obs, remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]:
        self.update_memory(obs)
        return self.symetric_action(
            self.sample_action(self.obs_to_tensor(obs), epsilon=0)
        )

    def get_actions_and_tensor(
        self, obs: Obs, update_memory=True
    ) -> tuple[
        np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]], torch.Tensor
    ]:
        if update_memory:
            self.update_memory(obs)
        tensor = self.obs_to_tensor(obs)
        return self.symetric_action(self.sample_action(tensor, epsilon=0)), tensor

    def symetric_action(self, action: PlayerAction) -> PlayerAction:
        if self.team_id == 0 or not self.symetric_player_1:
            return action

        action = action.copy()
        action[:, 0] = symetric_action(action[:, 0])
        return action

    @abstractmethod
    def _sample_action(self, obs_tensor: torch.Tensor, epsilon: float) -> PlayerAction:
        """Returns a numpy array of shape (batch_size, n_agents, 3) with the actions to take"""
        raise NotImplementedError

    @torch.no_grad()
    def sample_action(self, obs_tensor: torch.Tensor, epsilon: float) -> PlayerAction:
        return self._sample_action(obs_tensor, epsilon)

    def obs_to_tensor(self, obs: Obs) -> torch.Tensor:
        """! Warning ! This function does not update the memory"""
        return self.tensor_converter.convert(
            VecObs.from_obs(obs), np.array([self.team_id])
        ).squeeze(0)

    def save_net(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_net(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))


class BasicRLAgent(RLAgent):
    def __init__(
        self,
        player: str,
        env_params: EnvParams,
        device: str,
        model: nn.Module,
        mixte_strategy: bool = False,
    ) -> None:
        self.mixte_strategy = mixte_strategy

        super().__init__(
            player=player,
            env_params=env_params,
            device=device,
            model=model,
            tensor_converter=BasicTensorConverter(1),
            reward_shaper=GreedyRewardShaper() + DistanceToNearestRelicRewardShaper(),
            symetric_player_1=True,
        )

    def _sample_action(self, obs_tensor: torch.Tensor, epsilon: float) -> PlayerAction:
        # ^ WARNING ^ : This function does not use the sap action (it only moves the units)
        actions = torch.zeros((16, 3), dtype=torch.int32)

        if random.random() < epsilon:
            actions[:, 0] = random.randint(0, 4)
            return actions.numpy()

        out: torch.Tensor = (
            self.model(obs_tensor.unsqueeze(0).to(self.device)).cpu().squeeze(0)
        )
        if self.mixte_strategy:
            actions[:, 0] = (
                torch.multinomial(torch.softmax(out, dim=1), 1).squeeze(-1).int()
            )
        else:
            actions[:, 0] = out.argmax(1).int()

        return actions.numpy()
