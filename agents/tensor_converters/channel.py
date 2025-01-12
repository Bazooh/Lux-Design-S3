from typing import cast
from agents.obs import Obs
from agents.lux.utils import Tiles

from abc import ABC, abstractmethod
import numpy as np


class TensorChannels(ABC):
    names: list[str]
    n_channels: int

    def __init__(self):
        self.reset_memory()

    @abstractmethod
    def _convert(self, obs: Obs, team_id: int) -> np.ndarray:
        raise NotImplementedError

    def convert(self, obs: Obs, team_id: int) -> np.ndarray:
        return self._convert(obs, team_id)

    def reset_memory(self): ...

    def update_memory(self, obs: Obs, team_id: int): ...


class TensorChannel(TensorChannels):
    name: str
    n_channels = 1

    def __init__(self):
        self.names = [self.name]
        super().__init__()

    def convert(self, obs: Obs, team_id: int) -> np.ndarray:
        return self._convert(obs, team_id).reshape((1, 24, 24))


class SensorChannel(TensorChannel):
    name = "Sensor"

    def _convert(self, obs: Obs, team_id: int) -> np.ndarray:
        return ~obs.sensor_mask


class AsteroidChannel(TensorChannel):
    name = "Asteroid"

    def _convert(self, obs: Obs, team_id: int) -> np.ndarray:
        return obs.map_features.tile_type == Tiles.ASTEROID


class NebulaChannel(TensorChannel):
    name = "Nebula"

    def _convert(self, obs: Obs, team_id: int) -> np.ndarray:
        return obs.map_features.tile_type == Tiles.NEBULA


class RelicChannel(TensorChannel):
    name = "Relic"

    def reset_memory(self):
        self.discovered_relics_id: set[int] = set()
        self.discovered_all_relics: bool = False
        self.relic_tensor = np.zeros((24, 24), dtype=np.float32)

    def update_memory(self, obs: Obs, team_id: int):
        if self.discovered_all_relics:
            return

        discovered_this_frame_id = (
            set(obs.get_available_relics()) - self.discovered_relics_id
        )

        self.discovered_relics_id |= discovered_this_frame_id

        if len(discovered_this_frame_id) != 0:
            relic_nodes_discovered = obs.relic_nodes[list(discovered_this_frame_id)]
            self.relic_tensor[
                relic_nodes_discovered[:, 0], relic_nodes_discovered[:, 1]
            ] = 1

        if len(self.discovered_relics_id) == 6:
            self.discovered_all_relics = True

    def _convert(self, obs: Obs, team_id: int) -> np.ndarray:
        return self.relic_tensor


class EnergyChannel(TensorChannel):
    name = "Energy"

    def _convert(self, obs: Obs, team_id: int) -> np.ndarray:
        return obs.map_features.energy / 20


class EnemiesChannel(TensorChannel):
    name = "Enemies"

    def __init__(self, use_enery: bool = True):
        """If use_energy is True, the enemies will be represented by the energy they have. Otherwise, they will be represented by 1."""
        super().__init__()
        self.use_energy = use_enery

    def _convert(self, obs: Obs, team_id: int) -> np.ndarray:
        tensor = np.zeros((24, 24), dtype=np.float32)
        out = (obs.units.energy[1 - team_id] + 1) / 400 if self.use_energy else 1

        tensor[
            obs.units.position[1 - team_id, :, 0],
            obs.units.position[1 - team_id, :, 1],
        ] = out

        return tensor


class UnitChannel(TensorChannel):
    def __init__(self, unit_id: int, use_energy: bool = True):
        """If use_energy is True, the unit will be represented by the energy they have. Otherwise, they will be represented by 1."""
        self.name = f"Unit_{unit_id}"
        super().__init__()
        self.unit_id = unit_id
        self.use_energy = use_energy

    def _convert(self, obs: Obs, team_id: int) -> np.ndarray:
        tensor = np.zeros((24, 24))
        out = (obs.units.energy[team_id, self.unit_id] / 400) if self.use_energy else 1

        tensor[
            obs.units.position[team_id, :, 0],
            obs.units.position[team_id, :, 1],
        ] = out

        return tensor


class AllyUnitsChannels(TensorChannels):
    n_channels = 16
    names = [f"Units_{i}" for i in range(n_channels)]

    def __init__(self, use_enery: bool = True):
        """If use_energy is True, the unit will be represented by the energy they have. Otherwise, they will be represented by 1."""
        super().__init__()
        self.use_energy = use_enery

    def _convert(self, obs: Obs, team_id: int) -> np.ndarray:
        tensor = np.zeros((self.n_channels, 24, 24))
        out = (obs.units.energy[team_id] / 400) if self.use_energy else 1

        tensor[
            np.arange(16),
            obs.units.position[team_id, :, 0],
            obs.units.position[team_id, :, 1],
        ] = out

        return tensor


class RelicPointsChannels(TensorChannels):
    n_channels = 2
    names = ["Relics", "Points"]

    def __init__(self):
        self.relic_channel = RelicChannel()
        super().__init__()

    def reset_memory(self):
        self.last_team_points = 0
        self.unknown_relics_tensor = np.zeros((24, 24), dtype=np.float32)
        self.unknown_points_tensor = np.zeros((24, 24), dtype=np.float32)
        self.discovered_all_points = False
        self.relic_channel.reset_memory()

    def update_memory(self, obs: Obs, team_id: int):
        if self.discovered_all_points:
            return

        self.relic_channel.update_memory(obs, team_id)

        self.unknown_relics_tensor += (
            (self.unknown_relics_tensor == 0)
            * obs.sensor_mask
            * (2 * self.relic_channel.relic_tensor - 1)
        )

        team_points = cast(int, obs.team_points[team_id].item())
        points_gained = team_points - self.last_team_points
        self.last_team_points = team_points

        alive_units_id = obs.units_mask[team_id]
        if not alive_units_id.any():
            return

        # Cases surrounded by no relics -> no points
        unit_positions = np.array(obs.units.position[team_id])
        for unit_id in obs.get_available_units(team_id):
            unit_pos = unit_positions[unit_id]
            x, y = unit_pos[0].item(), unit_pos[1].item()

            if self.unknown_points_tensor[x, y] != 0:
                continue

            min_x = max(0, x - 2)
            max_x = min(24, x + 3)
            min_y = max(0, y - 2)
            max_y = min(24, y + 3)

            # If we are sure there are no relics around the unit, then this case earns no points
            if (self.unknown_relics_tensor[min_x:max_x, min_y:max_y] == -1).all():
                self.unknown_points_tensor[x, y] = -1

        alive_units_pos = obs.units.position[team_id][alive_units_id]
        unknown_points_mask = self.unknown_points_tensor[
            alive_units_pos[:, 0], alive_units_pos[:, 1]
        ]
        points_gained -= (unknown_points_mask == 1).sum().item()

        if points_gained == 0:
            self.unknown_points_tensor[
                alive_units_pos[:, 0], alive_units_pos[:, 1]
            ] -= (unknown_points_mask == 0).astype(np.int32)
        else:
            unknown_points_mask_is_unknown = (unknown_points_mask == 0).astype(np.int32)
            if unknown_points_mask_is_unknown.sum().item() == points_gained:
                self.unknown_points_tensor[
                    alive_units_pos[:, 0], alive_units_pos[:, 1]
                ] += unknown_points_mask_is_unknown

        if (self.unknown_points_tensor != 0).all():
            self.discovered_all_points = True

    def _convert(self, obs: Obs, team_id: int) -> np.ndarray:
        return np.stack(
            (self.relic_channel.convert(obs, team_id)[0], self.unknown_points_tensor)
        )
