from abc import abstractmethod, ABC
from typing import cast
import numpy as np
import torch

from agents.obs import Obs


class Memory(ABC):
    def __init__(self):
        """Don't override this method. Use _reset instead."""
        self.reset()

    @abstractmethod
    def _update(self, obs: Obs, team_id: int): ...

    @abstractmethod
    def _expand(self, obs: Obs, team_id: int) -> Obs: ...

    @abstractmethod
    def _reset(self): ...

    def reset(self):
        self._reset()

    def update(self, obs: Obs, team_id: int):
        self._update(obs, team_id)

    def expand(self, obs: Obs, team_id: int) -> Obs:
        return self._expand(obs, team_id)


class RelicMemory(Memory):
    def _reset(self):
        self.discovered_relics_id: set[int] = set()
        self.discovered_relics_id_list: list[int] = []
        self.relic_positions = -np.ones((6, 2), dtype=np.int32)
        self.discovered_all_relics = False
        self.discovered_this_frame_id: set[int] = set()
        self.relic_tensor = torch.zeros((24, 24), dtype=torch.float32)

    def _update(self, obs: Obs, team_id: int):
        if self.discovered_all_relics:
            return

        self.discovered_this_frame_id = (
            set(obs.get_available_relics()) - self.discovered_relics_id
        )

        relic_nodes = np.array(obs.relic_nodes)
        for relic_id in self.discovered_this_frame_id:
            self.discovered_relics_id.add(relic_id)
            self.discovered_relics_id_list.append(relic_id)
            self.relic_positions[relic_id] = relic_nodes[relic_id]

        if len(self.discovered_this_frame_id) != 0:
            discovered_this_frame_id_list = list(self.discovered_this_frame_id)
            relic_nodes_discovered = relic_nodes[discovered_this_frame_id_list]
            self.relic_tensor[
                relic_nodes_discovered[:, 0], relic_nodes_discovered[:, 1]
            ] = 1

        if len(self.discovered_relics_id) == 6:
            self.discovered_all_relics = True

    def _expand(self, obs: Obs, team_id: int) -> Obs:
        obs = Obs(
            units=obs.units,
            units_mask=obs.units_mask,
            sensor_mask=obs.sensor_mask,
            map_features=obs.map_features,
            relic_nodes=self.relic_positions,
            relic_nodes_mask=np.array(self.discovered_relics_id_list),
            team_points=obs.team_points,
            team_wins=obs.team_wins,
            steps=obs.steps,
            match_steps=obs.match_steps,
        )

        return obs


class RelicPointMemory(RelicMemory):
    def _reset(self):
        super()._reset()
        self.relic_points = np.zeros((6, 2), dtype=np.int32)
        self.last_team_points: int = 0
        self.unknown_relics_tensor = torch.zeros((24, 24), dtype=torch.float32)
        self.unknown_points_tensor = torch.zeros((24, 24), dtype=torch.float32)
        self.discovered_all_points = False

    def _update(self, obs: Obs, team_id: int):
        if self.discovered_all_points:
            return

        super()._update(obs, team_id)

        self.unknown_relics_tensor += (
            (self.unknown_relics_tensor == 0)
            * obs.sensor_mask
            * (2 * self.relic_tensor - 1)
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
            ] -= (unknown_points_mask == 0).int()
        else:
            unknown_points_mask_is_unknown = (unknown_points_mask == 0).int()
            if unknown_points_mask_is_unknown.sum().item() == points_gained:
                self.unknown_points_tensor[
                    alive_units_pos[:, 0], alive_units_pos[:, 1]
                ] += unknown_points_mask_is_unknown

        if (self.unknown_points_tensor != 0).all():
            self.discovered_all_points = True
