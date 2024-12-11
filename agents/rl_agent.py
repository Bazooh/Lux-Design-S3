import numpy as np
import torch
from lux.utils import print_debug
from models.dense import CNN
from tensors.tensor import TensorConverter
from lux.observation import Observation
from base_agent import Agent, N_Actions


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
        self, obs: Observation, remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[int, N_Actions], np.dtype[np.int32]]:
        result = self.model(self.tensor_converter.convert(obs))
        
        print_debug(result.shape)

        actions = np.zeros((self.env_cfg.max_units, 3), dtype=np.int32)
        actions[:, 0] = result.argmax(dim=1).numpy()

        return actions


class BasicRLAgent(RLAgent):
    def __init__(self, player: str, env_cfg: dict[str, int]) -> None:
        super().__init__(player, env_cfg, CNN(), TensorConverter())
