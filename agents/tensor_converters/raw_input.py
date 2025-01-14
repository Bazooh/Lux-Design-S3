from agents.obs import Obs

from abc import ABC, abstractmethod
import numpy as np


class TensorRawInputs(ABC):
    names: list[str]
    n_channels: int

    def __init__(self):
        self.reset_memory()

    def reset_memory(self): ...

    def update_memory(self, obs: Obs, team_id: int): ...

    def convert(self, obs: Obs, team_id: int):
        return self._convert(obs, team_id)

    @abstractmethod
    def _convert(self, obs: Obs, team_id: int):
        raise NotImplementedError


class TensorRawInput(TensorRawInputs):
    name: str
    n_channels = 1

    def __init__(self):
        self.names = [self.name]
        super().__init__()


class StepsLeftRawInput(TensorRawInput):
    name = "StepsLeft"

    def _convert(self, obs: Obs, team_id: int):
        return np.array((500 - obs.steps) / 500)


class MatchStepsLeftRawInput(TensorRawInput):
    name = "MatchStepsLeft"

    def _convert(self, obs: Obs, team_id: int):
        return np.array((100 - obs.match_steps) / 100)


class TeamPointsRawInput(TensorRawInput):
    name = "TeamPoints"

    def _convert(self, obs: Obs, team_id: int):
        return obs.team_points[team_id] / 100


class EnemyPointsRawInput(TensorRawInput):
    name = "EnemyPoints"

    def _convert(self, obs: Obs, team_id: int):
        return obs.team_points[1 - team_id] / 100


class PointsRawInput(TensorRawInput):
    name = "Points"

    def _convert(self, obs: Obs, team_id: int):
        return (
            np.concatenate(obs.team_points[team_id], obs.team_points[1 - team_id]) / 100
        )


class UnitsEnergyRawInput(TensorRawInput):
    name = "UnitsEnergy"

    def _convert(self, obs: Obs, team_id: int):
        return obs.units.energy[team_id] / 400
