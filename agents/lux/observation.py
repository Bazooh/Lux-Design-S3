import numpy as np
from typing import Any

from lux.utils import Vector2, Tiles


class PartialObservation:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask

    def avaible_ids(self):
        return np.where(self.mask)[0]

    def _avaible(self, property: np.ndarray) -> np.ndarray:
        return property[self.avaible_ids()]


class UnitsObservation(PartialObservation):
    def __init__(
        self, positions: np.ndarray, energies: np.ndarray, units_mask: np.ndarray
    ) -> None:
        super().__init__(units_mask)
        self.positions = positions
        self.energies = energies

    def get_position_of(self, team: int, unit_id: int) -> Vector2:
        return self.positions[team, unit_id]

    def get_energy_of(self, team: int, unit_id: int) -> int:
        return self.energies[team, unit_id]

    def avaible_positions(self):
        return self._avaible(self.positions)

    def avaible_energies(self):
        return self._avaible(self.energies)


class MapObservation(PartialObservation):
    def __init__(self, map_features: dict[str, Any], sensor_mask: np.ndarray) -> None:
        super().__init__(sensor_mask)
        self.energies = map_features["energy"]
        self.tiles = map_features["tile_type"]

    def get_energy_at(self, x: int, y: int) -> int:
        return self.energies[x, y]

    def get_tile_at(self, x: int, y: int) -> Tiles:
        return self.tiles[x, y]

    def avaible_tiles(self):
        return self._avaible(self.tiles)

    def avaible_energies(self):
        return self._avaible(self.energies)


class RelicObservation(PartialObservation):
    def __init__(self, relic_nodes: np.ndarray, relic_nodes_mask: np.ndarray) -> None:
        super().__init__(relic_nodes_mask)
        self.positions = relic_nodes

    def get_position_of(self, relic_id: int) -> Vector2:
        return self.positions[relic_id]

    def avaible_positions(self):
        return self._avaible(self.positions)


class Observation:
    def __init__(self, observation: dict[str, Any], team_id: int, step: int) -> None:
        self.team_id = team_id
        self.opposite_team_id = 1 - team_id

        self.allied_units = UnitsObservation(
            observation["units"]["position"][self.team_id],
            observation["units"]["energy"][self.team_id],
            observation["units_mask"][self.team_id],
        )
        self.opposite_units = UnitsObservation(
            observation["units"]["position"][self.opposite_team_id],
            observation["units"]["energy"][self.opposite_team_id],
            observation["units_mask"][self.opposite_team_id],
        )

        self.map = MapObservation(
            observation["map_features"], observation["sensor_mask"]
        )
        self.relics = RelicObservation(
            observation["relic_nodes"], observation["relic_nodes_mask"]
        )
        self.team_points: np.ndarray = observation["team_points"]
        self.team_wins: np.ndarray = observation["team_wins"]
        self.step: int = step
        self.steps: int = observation[
            "steps"
        ]  # No idea if there is a difference between the twos
        self.match_steps: int = observation["match_steps"]
