from abc import abstractmethod, ABC
from typing import cast
import numpy as np
import torch

from jax.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

from luxai_s3.state import EnvObs


class Memory(ABC):
    def __init__(self):
        """Don't override this method. Use _reset instead."""
        self.reset()

    @abstractmethod
    def _update(self, obs: EnvObs, team_id: int): ...

    @abstractmethod
    def _expand(self, obs: EnvObs, team_id: int) -> EnvObs: ...

    @abstractmethod
    def _reset(self): ...

    def reset(self):
        self._reset()

    def update(self, obs: EnvObs, team_id: int):
        self._update(obs, team_id)

    def expand(self, obs: EnvObs, team_id: int) -> EnvObs:
        return self._expand(obs, team_id)


class RelicMemory(Memory):
    def _reset(self):
        self.discovered_relics_id: set[int] = set()
        self.discovered_relics_id_list: list[int] = []
        self.relic_positions = -np.ones((6, 2), dtype=np.int32)
        self.discovered_all_relics = False
        self.discovered_this_frame_id: set[int] = set()
        self.relic_tensor = torch.zeros((24, 24), dtype=torch.float32)

    def _update(self, obs: EnvObs, team_id: int):
        if self.discovered_all_relics:
            return

        self.discovered_this_frame_id = (
            set(obs.get_avaible_relics()) - self.discovered_relics_id
        )

        for relic_id in self.discovered_this_frame_id:
            self.discovered_relics_id.add(relic_id)
            self.discovered_relics_id_list.append(relic_id)
            self.relic_positions[relic_id] = obs.relic_nodes[relic_id]

        if len(self.discovered_this_frame_id) != 0:
            discovered_this_frame_id_list = list(self.discovered_this_frame_id)
            relic_nodes = np.array(obs.relic_nodes)[discovered_this_frame_id_list]
            self.relic_tensor[relic_nodes[:, 0], relic_nodes[:, 1]] = 1

        if len(self.discovered_relics_id) == 6:
            self.discovered_all_relics = True

    def _expand(self, obs: EnvObs, team_id: int) -> EnvObs:
        obs = EnvObs(
            units=obs.units,
            units_mask=obs.units_mask,
            sensor_mask=obs.sensor_mask,
            map_features=obs.map_features,
            relic_nodes=self.relic_positions,
            relic_nodes_mask=self.discovered_relics_id_list,
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

    def _update(self, obs: EnvObs, team_id: int):
        if self.discovered_all_points:
            return

        super()._update(obs, team_id)

        self.unknown_relics_tensor += (
            (self.unknown_relics_tensor == 0)
            * from_dlpack(to_dlpack(obs.sensor_mask))
            * (2 * self.relic_tensor - 1)
        )

        team_points = cast(int, obs.team_points[team_id].item())
        points_gained = team_points - self.last_team_points
        self.last_team_points = team_points

        unit_mask = obs.units_mask[team_id]
        if not unit_mask.any():
            return

        unit_pos_mask = from_dlpack(
            to_dlpack(obs.units.position[team_id][unit_mask])
        ).to(torch.int32)
        unknown_points_mask = self.unknown_points_tensor[
            unit_pos_mask[:, 0], unit_pos_mask[:, 1]
        ]
        points_gained -= (unknown_points_mask == 1).sum().item()

        # Cases surrounded by no relics -> no points
        unit_positions = obs.units.position[team_id]
        for unit_id in obs.get_avaible_units(team_id):
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

        if points_gained == 0:
            self.unknown_points_tensor[unit_pos_mask[:, 0], unit_pos_mask[:, 1]] -= (
                unknown_points_mask == 0
            ).to(torch.int32)
        else:
            unknown_points_mask_is_unknown = (unknown_points_mask == 0).to(torch.int32)
            if unknown_points_mask_is_unknown.sum().item() == points_gained:
                self.unknown_points_tensor[
                    unit_pos_mask[:, 0], unit_pos_mask[:, 1]
                ] += unknown_points_mask_is_unknown

        if (self.unknown_points_tensor != 0).all():
            self.discovered_all_points = True
