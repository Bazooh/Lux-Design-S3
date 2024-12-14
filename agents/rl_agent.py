import numpy as np
import torch
from luxai_s3.state import EnvObs
from models.dense import CNN
from tensors.tensor import TensorConverter
from base_agent import Agent, N_Actions, N_Agents


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
        result = self.model(self.tensor_converter.convert(obs))

        actions = np.zeros((self.env_cfg.max_units, 3), dtype=np.int32)
        actions[:, 0] = result.argmax(dim=1).numpy()

        return actions

    def forward(self, obs: EnvObs) -> torch.Tensor:
        return self.model(self.tensor_converter.convert(obs))

    def sample_action(self, obs: EnvObs, epsilon: float):
        out = self.forward(obs)
        mask = torch.rand((out.shape[0],)) <= epsilon
        action = torch.empty(
            (
                out.shape[0],
                out.shape[1],
            )
        )
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        return action


class BasicRLAgent(RLAgent):
    def __init__(self, player: str, env_cfg: dict[str, int], model: CNN | None) -> None:
        super().__init__(
            player, env_cfg, model if model is not None else CNN(), TensorConverter()
        )
