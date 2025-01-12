from agents.reward_shapers.vec_reward import (
    VecGreedyRewardShaper,
    VecDistanceToNearestRelicRewardShaper,
)

import torch
from torch import nn


class VecBasicRLAgent:
    def __init__(
        self,
        n_envs: int,
        device: str,
        model: nn.Module,
        mixte_strategy: bool = False,
    ) -> None:
        self.n_envs = n_envs
        self.device = device
        self.model = model
        self.mixte_strategy = mixte_strategy
        self.reward_shaper = VecGreedyRewardShaper(
            n_envs
        ) + VecDistanceToNearestRelicRewardShaper(n_envs)

    @torch.no_grad()
    def sample_actions(self, obs_tensors: torch.Tensor, epsilon: float) -> torch.Tensor:
        # ^ WARNING ^ : This function does not use the sap action (it only moves the units)
        batch_size = obs_tensors.shape[0]

        mask = torch.rand(batch_size) < epsilon
        actions = torch.zeros((batch_size, 16, 3), dtype=torch.int32)

        if mask.all():
            actions[:, :, 0] = torch.randint(
                0, 5, actions[:, :, 0].shape[:2], dtype=torch.int32
            )
            return actions

        out: torch.Tensor = self.model(obs_tensors[~mask].to(self.device)).cpu()

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

        return actions
