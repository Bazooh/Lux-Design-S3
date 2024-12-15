import numpy as np
import torch
from luxai_s3.state import EnvObs
from agents.models.dense import CNN
from agents.tensors.tensor import TensorConverter
from agents.base_agent import Agent, N_Actions, N_Agents


class RLAgent(Agent):
    def __init__(
        self,
        player: str,
        env_cfg: dict[str, int],
        model: torch.nn.Module,
        tensor_converter: TensorConverter,
    ) -> None:
        super().__init__(player, env_cfg)
        self.model = model
        self.tensor_converter = tensor_converter

    def actions(
        self, obs: EnvObs, remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]:
        result = self.model(self.tensor_converter.convert(obs, self.team_id))

        actions = np.zeros((self.env_cfg.max_units, 3), dtype=np.int32)
        actions[:, 0] = result.argmax(dim=1).numpy()

        return actions

    def sample_action(self, obs_tensor: torch.Tensor, epsilon: float):
        """Returns a tensor of shape (batch_size, n_agents, 3) with the actions to take"""
        # ! WARNING ! : This function does not use the sap action (it only moves the units)
        out = self.model(obs_tensor)

        batch_size, n_agents, n_actions = out.shape

        mask = torch.rand((batch_size,)) <= epsilon
        action = torch.zeros((batch_size, n_agents, 3), dtype=torch.int32)
        action[mask, :, 0] = torch.randint(0, n_actions, action[mask].shape[:2]).int()
        action[~mask, :, 0] = out[~mask].argmax(dim=2).int()
        return action

    def obs_to_tensor(self, obs: EnvObs) -> torch.Tensor:
        return self.tensor_converter.convert(obs, self.team_id)


class BasicRLAgent(RLAgent):
    def __init__(self, player: str, env_cfg: dict[str, int], model: CNN | None) -> None:
        super().__init__(
            player, env_cfg, model if model is not None else CNN(), TensorConverter()
        )
