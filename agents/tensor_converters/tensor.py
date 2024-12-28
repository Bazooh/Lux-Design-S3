from abc import ABC, abstractmethod
import numpy as np
import torch
from agents.lux.utils import Tiles
from agents.memory.memory import Memory, RelicMemory, RelicPointMemory
from src.luxai_s3.state import EnvObs
from jax.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack


class TensorConverter(ABC):
    """
    Abstract base class for converting observations into tensor representations.
    """

    def __init__(self):
        self.channel_names = []

    @abstractmethod
    def convert(
        self,
        obs: EnvObs,
        team_id: int,
        symetric_player_1: bool = True,
        memory: Memory | None = None,
    ) -> torch.Tensor:
        """
        Convert an observation into a tensor representation.

        Args:
            obs (EnvObs): The environment observation.
            team_id (int): The team identifier.

        Returns:
            torch.Tensor: The converted tensor representation.
        """
        pass


class BasicMapExtractor(TensorConverter):
    def __init__(self):
        super().__init__()
        self.channel_names = [
            "Unknown",
            "Asteroid",
            "Nebula",
            "Relic",
            "Points",
            "Energy_Field",
            "Enemy_Units",
        ]
        self.channel_names += [f"Ally_Unit_{i}" for i in range(1, 17)]

    def convert(
        self,
        obs: EnvObs,
        team_id: int,
        symetric_player_1: bool = True,
        memory: Memory | None = None,
    ) -> torch.Tensor:
        """
        Shape : (22, width, height)

        0: Unknown    (0: Known, 1: Unknown)
        1: Asteroid   (0: Empty or unknown, 1: Asteroid)
        2: Nebula     (0: Empty or unknown, 1: Nebula)
        3: Relic      (0: Empty or unknown, 1: Relic)
        4: Points     (0: Unknown, 1: Points, -1: No points)
        5: Energy     (0-1: Energy amount / max_unit_energy)
        6: Enemy      (0-max_units: Sum enemy unit energy / max_unit_energy)
        7 - 22: Units (0-1: Unit energy / max_unit_energy)
        """
        # device = str(obs.map_features.tile_type.device)
        device = "cpu"

        tensor = torch.zeros(
            23,
            obs.map_features.energy.shape[0],
            obs.map_features.energy.shape[1],
            dtype=torch.float32,
            device=device,
        )

        tensor[0] = ~from_dlpack(to_dlpack(obs.sensor_mask))
        tensor[1] = from_dlpack(to_dlpack(obs.map_features.tile_type)) == Tiles.ASTEROID
        tensor[2] = from_dlpack(to_dlpack(obs.map_features.tile_type)) == Tiles.NEBULA

        if isinstance(memory, RelicMemory):
            tensor[3] = memory.relic_tensor
        else:
            relic_nodes = np.array(obs.relic_nodes[obs.relic_nodes_mask])
            tensor[3, relic_nodes[:, 0], relic_nodes[:, 1]] = 1

        if isinstance(memory, RelicPointMemory):
            tensor[4] = memory.unknown_points_tensor

        tensor[5] = from_dlpack(to_dlpack(obs.map_features.energy)) / 20

        positions = np.array(obs.units.position)
        tensor[
            6,
            positions[1 - team_id, :, 0],
            positions[1 - team_id, :, 1],
        ] = (from_dlpack(to_dlpack(obs.units.energy[1 - team_id])) + 1) / 400

        tensor[
            torch.arange(7, 23),
            positions[team_id, :, 0],
            positions[team_id, :, 1],
        ] = (from_dlpack(to_dlpack(obs.units.energy[team_id])) + 1) / 400

        return tensor.flip(1, 2) if symetric_player_1 and team_id == 1 else tensor
