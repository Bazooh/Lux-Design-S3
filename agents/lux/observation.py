from colorama import Back, Fore, Style
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

    def get_position_of(self, unit_id: int) -> Vector2:
        return self.positions[unit_id]

    def get_energy_of(self, unit_id: int) -> int:
        return self.energies[unit_id]

    def avaible_positions(self):
        return self._avaible(self.positions)

    def avaible_energies(self):
        return self._avaible(self.energies)

    def get_units_id_at(self, x: int, y: int):
        return np.where((self.positions[0] == x) & (self.positions[1] == y))[0]


class MapObservation(PartialObservation):
    def __init__(self, map_features: dict[str, Any], sensor_mask: np.ndarray) -> None:
        super().__init__(sensor_mask)
        self.energies = map_features["energy"]
        self.tiles = map_features["tile_type"]
        self.height = self.tiles.shape[0]
        self.width = self.tiles.shape[1]

    def get_energy_at(self, x: int, y: int) -> int:
        return self.energies[x, y]

    def get_tile_at(self, x: int, y: int) -> Tiles:
        return self.tiles[x, y]

    def avaible_tiles(self):
        return self._avaible(self.tiles)

    def avaible_energies(self):
        return self._avaible(self.energies)

    def __str__(self) -> str:
        tiles_symbol = {
            Tiles.UNKNOWN: Back.BLACK + "?",
            Tiles.EMPTY: Back.BLACK + " ",
            Tiles.NEBULA: Back.MAGENTA + " ",
            Tiles.ASTEROID: Back.WHITE + " ",
        }

        string = "Map :\n\n"
        for i in range(self.tiles.shape[0]):
            for j in range(self.tiles.shape[1]):
                string += f"{tiles_symbol[self.get_tile_at(j, i)]}"
            string += Style.RESET_ALL + "\n"
        return string


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

    def get_n_units_at(self, team: int, x: int, y: int) -> int:
        if team == self.team_id:
            return np.sum(
                (self.allied_units.positions[0] == x)
                & (self.allied_units.positions[1] == y)
            )
        else:
            return np.sum(
                (self.opposite_units.positions[0] == x)
                & (self.opposite_units.positions[1] == y)
            )

    def get_team_points(self):
        return self.team_points[self.team_id]

    def get_team_wins(self):
        return self.team_wins[self.team_id]

    def get_opposite_team_points(self):
        return self.team_points[self.opposite_team_id]

    def get_opposite_team_wins(self):
        return self.team_wins[self.opposite_team_id]

    def __str__(self) -> str:
        string = f"Observation at step {self.step} :\n"
        string += f"Team {self.team_id} wins : {self.get_team_wins()} points : {self.get_team_points()}\n"
        string += f"Team {self.opposite_team_id} wins : {self.get_opposite_team_wins()} points : {self.get_opposite_team_points()}\n"

        tiles_color = {
            Tiles.UNKNOWN: Back.BLACK,
            Tiles.EMPTY: Back.BLACK,
            Tiles.NEBULA: Back.MAGENTA,
            Tiles.ASTEROID: Back.WHITE,
        }

        for y in range(self.map.height):
            for x in range(self.map.width):
                n_units = self.get_n_units_at(self.team_id, x, y) - self.get_n_units_at(
                    self.opposite_team_id, x, y
                )
                unit_color = (
                    Fore.GREEN
                    if n_units > 0
                    else Fore.RED
                    if n_units < 0
                    else Fore.BLACK
                )

                tile = self.map.get_tile_at(x, y)
                unit_char = str(abs(n_units)) if n_units != 0 else " "

                string += f"{tiles_color[tile]}{unit_color}{unit_char if tile != Tiles.UNKNOWN else '?'}"
            string += Style.RESET_ALL + "\n"
        return string
