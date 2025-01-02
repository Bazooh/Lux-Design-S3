from agents.tensor_converters.tensor import TensorConverter
from agents.obs import Obs
from agents.memory.memory import Memory
import torch


class IndependantTensorConverter(TensorConverter):
    def __init__(self, tensor_converter: TensorConverter):
        super().__init__()
        self.tensor_converter = tensor_converter

    def convert(
        self,
        obs: Obs,
        team_id: int,
        symetric_player_1: bool = True,
        memory: Memory | None = None,
    ) -> torch.Tensor:
        tensor = self.tensor_converter.convert(obs, team_id, symetric_player_1, memory)

        shared_obs = tensor[:-16]
        positions_sum = tensor[-16:].sum(dim=0)

        return torch.stack(
            [
                torch.cat(
                    (
                        shared_obs,
                        tensor[i - 16].unsqueeze(0),
                        (positions_sum - tensor[i - 16]).unsqueeze(0),
                    ),
                    dim=0,
                )
                for i in range(16)
            ]
        )
