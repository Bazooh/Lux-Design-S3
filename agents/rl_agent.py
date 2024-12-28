from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn
from agents.memory.memory import Memory, RelicPointMemory
from luxai_s3.env import PlayerAction
from agents.models.dense import CNN
from agents.tensor_converters.tensor import TensorConverter, BasicMapExtractor
from agents.reward_shapers.reward import (
    GreedyRewardShaper,
    RewardShaper,
)
from agents.base_agent import Agent, N_Actions, N_Agents
from agents.obs import Obs


def symetric_action_not_vectorized(action: int) -> int:
    return [0, 3, 4, 1, 2][action]


symetric_action = np.vectorize(symetric_action_not_vectorized)


class RLAgent(Agent):
    def __init__(
        self,
        player: str,
        env_cfg: dict[str, int],
        model: torch.nn.Module,
        tensor_converter: TensorConverter,
        reward_shaper: RewardShaper,
        memory: Memory | None = None,
        symetric_player_1: bool = True,
    ) -> None:
        super().__init__(player, env_cfg, memory)
        self.model = model
        self.tensor_converter = tensor_converter
        self.reward_shaper = reward_shaper
        self.symetric_player_1 = symetric_player_1

    def _actions(
        self, obs: Obs, remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]:
        return self.symetric_action(
            self.sample_action(self.obs_to_tensor(obs), epsilon=0)
        )

    def get_actions_and_tensor(
        self, obs: Obs, update_memory=True
    ) -> tuple[
        np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]], torch.Tensor
    ]:
        if update_memory:
            self.update_obs(obs)
        tensor = self.obs_to_tensor(obs)
        return self.symetric_action(self.sample_action(tensor, epsilon=0)), tensor

    def symetric_action(self, action: PlayerAction) -> PlayerAction:
        if self.team_id == 0 or not self.symetric_player_1:
            return action

        action = action.copy()
        action[:, 0] = symetric_action(action[:, 0])
        return action

    @abstractmethod
    def sample_action(self, obs_tensor: torch.Tensor, epsilon: float) -> PlayerAction:
        """Returns a numpy array of shape (n_agents, 3) with the actions to take"""
        raise NotImplementedError

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
        env_cfg: dict[str, int],
        model: nn.Module | None = None,
        model_path: str | None = None,
    ) -> None:
        assert (
            model is None or model_path is None
        ), "Only one of model or model_path can be provided"

        if model_path is not None:
            model = CNN()
            model.load_state_dict(torch.load(model_path))

        super().__init__(
            player=player,
            env_cfg=env_cfg,
            model=model if model is not None else CNN(),
            tensor_converter=BasicMapExtractor(),
            reward_shaper=GreedyRewardShaper(),
            memory=RelicPointMemory(),
            symetric_player_1=True,
        )

    def sample_action(self, obs_tensor: torch.Tensor, epsilon: float) -> PlayerAction:
        # ! WARNING ! : This function does not use the sap action (it only moves the units)
        out = self.model(obs_tensor)

        batch_size, n_agents, n_actions = out.shape

        mask = torch.rand((batch_size,)) <= epsilon
        action = torch.zeros((batch_size, n_agents, 3), dtype=torch.int32)
        action[mask, :, 0] = torch.randint(0, n_actions, action[mask].shape[:2]).int()
        action[~mask, :, 0] = out[~mask].argmax(dim=2).int()
        return action.squeeze(0).detach().numpy()
