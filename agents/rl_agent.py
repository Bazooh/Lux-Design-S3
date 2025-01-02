from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn
from agents.memory.memory import Memory
from luxai_s3.env import PlayerAction
from agents.tensor_converters.tensor import TensorConverter, MinimalTensorConverter
from agents.reward_shapers.reward import (
    GreedyExploreRewardShaper,
    ExploreRewardShaper,
    RewardShaper,
)
from agents.base_agent import Agent, N_Actions, N_Agents
from agents.obs import EnvParams, Obs


def symetric_action_not_vectorized(action: int) -> int:
    return [0, 3, 4, 1, 2][action]


symetric_action = np.vectorize(symetric_action_not_vectorized)


class RLAgent(Agent):
    def __init__(
        self,
        player: str,
        env_params: EnvParams,
        device: str,
        model: torch.nn.Module,
        tensor_converter: TensorConverter,
        reward_shaper: RewardShaper,
        memory: Memory | None = None,
        symetric_player_1: bool = True,
    ) -> None:
        super().__init__(player, env_params, memory)
        self.device = device
        self.model = model
        self.tensor_converter = tensor_converter
        self.reward_shaper = reward_shaper
        self.symetric_player_1 = symetric_player_1

    def _actions(
        self, obs: Obs, remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]:
        return self.symetric_action(
            self.sample_one_action(self.obs_to_tensor(obs), epsilon=0)
        )

    def get_actions_and_tensor(
        self, obs: Obs, update_memory=True
    ) -> tuple[
        np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]], torch.Tensor
    ]:
        if update_memory:
            self.update_obs(obs)
        tensor = self.obs_to_tensor(obs)
        return self.symetric_action(self.sample_one_action(tensor, epsilon=0)), tensor

    def symetric_action(self, action: PlayerAction) -> PlayerAction:
        if self.team_id == 0 or not self.symetric_player_1:
            return action

        action = action.copy()
        action[:, 0] = symetric_action(action[:, 0])
        return action

    @abstractmethod
    def _sample_action(self, obs_tensor: torch.Tensor, epsilon: float) -> PlayerAction:
        """Returns a numpy array of shape (n_agents, 3) with the actions to take"""
        raise NotImplementedError

    @torch.no_grad()
    def sample_action(self, obs_tensor: torch.Tensor, epsilon: float) -> PlayerAction:
        return self._sample_action(obs_tensor, epsilon)

    def sample_one_action(
        self, obs_tensor: torch.Tensor, epsilon: float
    ) -> PlayerAction:
        return self.sample_action(obs_tensor.unsqueeze(0), epsilon).squeeze(0)

    def obs_to_tensor(self, obs: Obs) -> torch.Tensor:
        """! Warning ! This function does not update the memory"""
        return self.tensor_converter.convert(
            self.expand_obs(obs), self.team_id, self.symetric_player_1, self.memory
        )

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
            tensor_converter=MinimalTensorConverter(),
            reward_shaper=GreedyExploreRewardShaper(),
            memory=None,
            symetric_player_1=True,
        )

    def _sample_action(self, obs_tensor: torch.Tensor, epsilon: float) -> PlayerAction:
        # ^ WARNING ^ : This function does not use the sap action (it only moves the units)
        batch_size = obs_tensor.shape[0]

        mask = torch.rand(batch_size) < epsilon
        actions = torch.zeros((batch_size, 16, 3), dtype=torch.int32)

        if mask.all():
            actions[:, :, 0] = torch.randint(
                0, 5, actions[:, :, 0].shape[:2], dtype=torch.int32
            )
            return actions.numpy()

        out: torch.Tensor = self.model(obs_tensor[~mask].to(self.device)).cpu()

        actions[mask, :, 0] = torch.randint(
            0, 5, actions[mask, :, 0].shape[:2], dtype=torch.int32
        )
        if self.mixte_strategy:
            actions[~mask, :, 0] = (
                torch.multinomial(torch.softmax(out, dim=2).view(-1, 5), 1)
                .squeeze(-1)
                .int()
                .view(-1, 16)
            )
        else:
            actions[~mask, :, 0] = out.argmax(2).int()

        return actions.numpy()
