import torch
from torch import nn
from env_interface import TensorInfo


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

    @torch.no_grad()
    def sample_actions(
        self, obs: dict[TensorInfo, torch.Tensor], epsilon: float
    ) -> torch.Tensor:
        # ^ WARNING ^ : This function does not use the sap action (it only moves the units)
        batch_size = obs["channels"].shape[0] * 2

        mask = torch.rand(batch_size) < epsilon
        actions = torch.zeros((batch_size, 16, 3), dtype=torch.int32)

        if mask.all():
            actions[:, :, 0] = torch.randint(
                0, 5, actions[:, :, 0].shape[:2], dtype=torch.int32
            )
            return actions

        out: torch.Tensor = self.model(
            obs["channels"].view(self.n_envs * 2, -1, 24, 24)[~mask].to(self.device),
            obs["raw_inputs"].view(self.n_envs * 2, -1)[~mask].to(self.device),
        ).cpu()

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
