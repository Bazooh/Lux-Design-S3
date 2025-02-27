from typing import Any
from dataclasses import dataclass
import numpy as np
PlayerName = Any

@dataclass(frozen=True)
class Units:
    position: np.ndarray
    """(team_id, unit_id, axis (x=0, y=1))"""
    energy: np.ndarray
    """(team_id, unit_id)"""


@dataclass(frozen=True)
class MapFeatures:
    energy: np.ndarray
    """(x, y)"""
    tile_type: np.ndarray
    """(x, y)"""


@dataclass(frozen=True)
class Obs:
    units: Units
    units_mask: np.ndarray
    """(team_id, unit_id)"""
    sensor_mask: np.ndarray
    """(x, y)"""
    map_features: MapFeatures
    relic_nodes: np.ndarray
    """(relic_id, axis (x=0, y=1))"""
    relic_nodes_mask: np.ndarray
    """(relic_id)"""
    team_points: np.ndarray
    """(team_id)"""
    team_wins: np.ndarray
    """(team_id)"""
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
class EnvParams:
    max_units: int
    match_count_per_episode: int
    max_steps_in_match: int
    map_height: int
    map_width: int
    num_teams: int
    unit_move_cost: int
    unit_sap_cost: int
    unit_sap_range: int
    unit_sensor_range: int

    @staticmethod
    def from_dict(env_params: dict[str, Any]):
        return EnvParams(
            max_units=env_params["max_units"],
            match_count_per_episode=env_params["match_count_per_episode"],
            max_steps_in_match=env_params["max_steps_in_match"],
            map_height=env_params["map_height"],
            map_width=env_params["map_width"],
            num_teams=env_params["num_teams"],
            unit_move_cost=env_params["unit_move_cost"],
            unit_sap_cost=env_params["unit_sap_cost"],
            unit_sap_range=env_params["unit_sap_range"],
            unit_sensor_range=env_params["unit_sensor_range"],
        )


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


@dataclass(frozen=True)
class VecUnits:
    position: np.ndarray
    """(n_envs, team_id, unit_id, axis (x=0, y=1))"""
    energy: np.ndarray
    """(n_envs, team_id, unit_id)"""


@dataclass(frozen=True)
class VecMapFeatures:
    energy: np.ndarray
    """(n_envs, x, y)"""
    tile_type: np.ndarray
    """(n_envs, x, y)"""


@dataclass(frozen=True)
class VecObs:
    units: VecUnits
    units_mask: np.ndarray
    """(n_envs, team_id, unit_id)"""
    sensor_mask: np.ndarray
    """(n_envs, x, y)"""
    map_features: VecMapFeatures
    relic_nodes: np.ndarray
    """(n_envs, relic_id, axis (x=0, y=1))"""
    relic_nodes_mask: np.ndarray
    """(n_envs, relic_id)"""
    team_points: np.ndarray
    """(n_envs, team_id)"""
    team_wins: np.ndarray
    """(n_envs, team_id)"""
    steps: np.ndarray
    """(n_envs,)"""
    match_steps: np.ndarray
    """(n_envs,)"""

    @staticmethod
    def from_dict(obs: dict[str, Any]):
        return VecObs(
            units=VecUnits(obs["units"]["position"], obs["units"]["energy"]),
            units_mask=obs["units_mask"],
            sensor_mask=obs["sensor_mask"],
            map_features=VecMapFeatures(
                obs["map_features"]["energy"], obs["map_features"]["tile_type"]
            ),
            relic_nodes=obs["relic_nodes"],
            relic_nodes_mask=obs["relic_nodes_mask"],
            team_points=obs["team_points"],
            team_wins=obs["team_wins"],
            steps=obs["steps"],
            match_steps=obs["match_steps"],
        )

    @staticmethod
    def from_obs(obs: Obs):
        return VecObs(
            units=VecUnits(
                np.expand_dims(obs.units.position, axis=0),
                np.expand_dims(obs.units.energy, axis=0),
            ),
            units_mask=np.expand_dims(obs.units_mask, axis=0),
            sensor_mask=np.expand_dims(obs.sensor_mask, axis=0),
            map_features=VecMapFeatures(
                np.expand_dims(obs.map_features.energy, axis=0),
                np.expand_dims(obs.map_features.tile_type, axis=0),
            ),
            relic_nodes=np.expand_dims(obs.relic_nodes, axis=0),
            relic_nodes_mask=np.expand_dims(obs.relic_nodes_mask, axis=0),
            team_points=np.expand_dims(obs.team_points, axis=0),
            team_wins=np.expand_dims(obs.team_wins, axis=0),
            steps=np.expand_dims(obs.steps, axis=0),
            match_steps=np.expand_dims(obs.match_steps, axis=0),
        )

    def get_available_relics(self, env_id: int):
        return np.where(self.relic_nodes_mask[env_id])[0]

    def get_available_units(self, env_id: int, team_id: int):
        return np.where(self.units_mask[env_id, team_id])[0]


@dataclass(frozen=True)
class VecGodObs:
    player_0: VecObs
    player_1: VecObs

    @staticmethod
    def from_dict(obs: dict[PlayerName, Any]):
        return VecGodObs(
            VecObs.from_dict(obs["player_0"]),
            VecObs.from_dict(obs["player_1"]),
        )


@dataclass(frozen=True)
class VecEnvParams:
    max_units: np.ndarray
    match_count_per_episode: np.ndarray
    max_steps_in_match: np.ndarray
    map_height: np.ndarray
    map_width: np.ndarray
    num_teams: np.ndarray
    unit_move_cost: np.ndarray
    unit_sap_cost: np.ndarray
    unit_sap_range: np.ndarray
    unit_sensor_range: np.ndarray

    @staticmethod
    def from_dict(env_params: dict[str, Any]):
        return VecEnvParams(
            max_units=env_params["max_units"],
            match_count_per_episode=env_params["match_count_per_episode"],
            max_steps_in_match=env_params["max_steps_in_match"],
            map_height=env_params["map_height"],
            map_width=env_params["map_width"],
            num_teams=env_params["num_teams"],
            unit_move_cost=env_params["unit_move_cost"],
            unit_sap_cost=env_params["unit_sap_cost"],
            unit_sap_range=env_params["unit_sap_range"],
            unit_sensor_range=env_params["unit_sensor_range"],
        )
