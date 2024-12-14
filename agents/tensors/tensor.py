import torch
from lux.utils import Tiles
from luxai_s3.state import EnvObs


class TensorConverter:
    def convert(self, obs: EnvObs):
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

        tensor = torch.zeros(23, obs.map.width, obs.map.height, dtype=torch.float32)

        tensor[0] = torch.tensor(~obs.map.mask, dtype=torch.float32)
        tensor[1] = torch.tensor(obs.map.tiles == Tiles.ASTEROID, dtype=torch.float32)
        tensor[2] = torch.tensor(obs.map.tiles == Tiles.NEBULA, dtype=torch.float32)
        for relic in obs.relics.avaible_positions():
            tensor[3, relic[0], relic[1]] = 1
        tensor[4] = torch.tensor(obs.map.energies / 400, dtype=torch.float32)
        for id in obs.opposite_units.avaible_ids():
            tensor[5] += torch.tensor(
                obs.opposite_units.get_energy_of(id) / 400, dtype=torch.float32
            )
        for id in obs.allied_units.avaible_ids():
            tensor[6 + id] = torch.tensor(
                obs.allied_units.get_energy_of(id) / 400, dtype=torch.float32
            )

        return tensor
