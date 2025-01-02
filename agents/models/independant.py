import torch
import torch.nn as nn


class IndependantModel(nn.Module):
    def __init__(self, model: nn.Module):
        """The model must have n_agents set to 1"""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_agents, n_channels, width, height = x.shape

        x = x.view(-1, n_channels, width, height)
        x = self.model(x)
        x = x.view(batch_size, n_agents, -1)

        return x
