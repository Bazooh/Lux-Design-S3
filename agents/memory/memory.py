from abc import abstractmethod, ABC
import numpy as np

from luxai_s3.state import EnvObs


class Memory(ABC):
    def __init__(self):
        self.reset()

    @abstractmethod
    def update(self, obs: EnvObs): ...

    @abstractmethod
    def expand(self, obs: EnvObs) -> EnvObs: ...

    @abstractmethod
    def reset(self): ...


class RelicMemory(Memory):
    def reset(self):
        self.discovered_relics_id: set[int] = set()
        self.discovered_relics_id_list: list[int] = []
        self.relic_positions: np.ndarray = -np.ones((6, 2), dtype=np.int32)
        self.discovered_all_relics = False

    def update(self, obs: EnvObs):
        if self.discovered_all_relics:
            return

        for relic_id in obs.get_avaible_relics():
            if relic_id not in self.discovered_relics_id:
                self.discovered_relics_id.add(relic_id)
                self.discovered_relics_id_list.append(relic_id)
                self.relic_positions[relic_id] = obs.relic_nodes[relic_id]

        if len(self.discovered_relics_id) == 6:
            self.discovered_all_relics = True

    def expand(self, obs: EnvObs) -> EnvObs:
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
