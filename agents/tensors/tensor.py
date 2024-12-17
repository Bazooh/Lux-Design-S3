import numpy as np
import torch
from agents.lux.utils import Tiles
from src.luxai_s3.state import EnvObs


class TensorConverter:
    def __init__(self):
        self.channel_names = [
            "Unknown",
            "Asteroid",
            "Nebula",
            "Relic",
            "Energy_Field",
            "Enemy_Units",
        ]
        self.channel_names += [f"Ally_Unit_{i}" for i in range(1, 17)]
        self.channel_names.append("Dummy")
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

        tensor[0] = torch.tensor(
            ~np.array(obs.sensor_mask, dtype=bool), dtype=torch.float32
        )
        tensor[1] = torch.tensor(
            np.array(obs.map_features.tile_type, dtype=np.int32) == Tiles.ASTEROID,
            dtype=torch.float32,
        )
        tensor[2] = torch.tensor(
            np.array(obs.map_features.tile_type, dtype=np.int32) == Tiles.NEBULA,
            dtype=torch.float32,
        )
        for relic_id in obs.get_avaible_relics():
            x = obs.relic_nodes[relic_id][0].item()
            y = obs.relic_nodes[relic_id][1].item()
            tensor[3, x, y] = 1
        tensor[4] = torch.tensor(
            np.array(obs.map_features.energy, dtype=np.float32) / 20,
            dtype=torch.float32,
        )
        for id in obs.get_avaible_units(1 - team_id):
            pos = obs.units.position[1 - team_id][id]
            x, y = pos[0].item(), pos[1].item()
            tensor[5, x, y] +=  float(obs.units.energy[1 - team_id][id] / 400)

        for id in obs.get_avaible_units(team_id):
            pos = obs.units.position[team_id][id]
            x, y = pos[0].item(), pos[1].item()
            tensor[6 + id, x, y] = float(obs.units.energy[team_id][id] / 400)

        return tensor
