from typing import Any
from dataclasses import dataclass
import numpy as np
from luxai_s3.env import PlayerName


@dataclass(frozen=True)
class Units:
    position: np.ndarray
    energy: np.ndarray


@dataclass(frozen=True)
class MapFeatures:
    energy: np.ndarray
    tile_type: np.ndarray


@dataclass(frozen=True)
class Obs:
    units: Units
    units_mask: np.ndarray
    sensor_mask: np.ndarray
    map_features: MapFeatures
    relic_nodes: np.ndarray
    relic_nodes_mask: np.ndarray
    team_points: np.ndarray
    team_wins: np.ndarray
    steps: int
    match_steps: int

    @staticmethod
    def from_dict(obs: dict[str, Any]):
        return Obs(
            units=Units(obs["units"]["position"], obs["units"]["energy"]),
            units_mask=obs["units_mask"],
            sensor_mask=obs["sensor_mask"],
            map_features=MapFeatures(
                obs["map_features"]["energy"], obs["map_features"]["tile_type"]
            ),
            relic_nodes=obs["relic_nodes"],
            relic_nodes_mask=obs["relic_nodes_mask"],
            team_points=obs["team_points"],
            team_wins=obs["team_wins"],
            steps=obs["steps"],
            match_steps=obs["match_steps"],
        )

    def get_available_units(self, team_id):
        return np.where(self.units_mask[team_id])[0]

    def get_available_relics(self):
        return np.where(self.relic_nodes_mask)[0]


@dataclass(frozen=True)
class GodObs:
    player_0: Obs
    player_1: Obs

    @staticmethod
    def from_dict(obs: dict[PlayerName, Any]):
        return GodObs(
            Obs.from_dict(obs["player_0"]),
            Obs.from_dict(obs["player_1"]),
        )
