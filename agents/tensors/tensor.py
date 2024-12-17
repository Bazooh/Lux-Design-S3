import numpy as np
import torch
from agents.lux.utils import Tiles
from src.luxai_s3.state import EnvObs
from jax.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack


class TensorConverter:
    def convert(self, obs: EnvObs, team_id: int) -> torch.Tensor:
        """
        Shape : (23, width, height)

        0: Unknown    (0: Known, 1: Unknown)
        1: Asteroid   (0: Empty or unknown, 1: Asteroid)
        2: Nebula     (0: Empty or unknown, 1: Nebula)
        3: Relic      (0: Empty or unknown, 1: Relic)
        4: Energy     (0-1: Energy amount / max_unit_energy)
        5: Enemy      (0-max_units: Sum enemy unit energy / max_unit_energy)
        6 - 22: Units (0-1: Unit energy / max_unit_energy)
        """

        tensor = torch.zeros(
            23,
            obs.map_features.energy.shape[0],
            obs.map_features.energy.shape[1],
            dtype=torch.float32,
        )

        tensor[0] = from_dlpack(to_dlpack(~obs.sensor_mask))
        tensor[1] = from_dlpack(to_dlpack(obs.map_features.tile_type == Tiles.ASTEROID))
        tensor[2] = from_dlpack(to_dlpack(obs.map_features.tile_type == Tiles.NEBULA))
        for relic_id in obs.get_avaible_relics():
            x = obs.relic_nodes[relic_id][0].item()
            y = obs.relic_nodes[relic_id][1].item()
            tensor[3, x, y] = 1
        tensor[4] = from_dlpack(to_dlpack(obs.map_features.energy / 20))
        for id in obs.get_avaible_units(1 - team_id):
            tensor[5] += from_dlpack(to_dlpack(obs.units.energy[1 - team_id][id] / 400))
        for id in obs.get_avaible_units(team_id):
            tensor[6 + id] = from_dlpack(to_dlpack(obs.units.energy[team_id][id] / 400))

        return tensor
