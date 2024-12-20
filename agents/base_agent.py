import abc
from typing import Any, Literal
from agents.lux.utils import Vector2
from agents.lux.env_config import EnvConfig
import numpy as np
import torch

from agents.memory.memory import Memory
from luxai_s3.state import EnvObs


N_Actions = Literal[3]
N_Agents = Literal[16]


class Agent:
    def __init__(
        self, player: str, env_cfg: dict[str, int], memory: Memory | None = None
    ) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = EnvConfig(env_cfg)
        self.memory = memory

        self.relic_node_positions: list[Vector2] = []
        self.discovered_relic_nodes_ids: set[int] = set()
        self.relic_tensor = torch.zeros((24, 24), dtype=torch.float32)

        self.unit_explore_locations: dict[int, Any] = (
            dict()
        )  # "Any" should be "Vector2" but for some reason they put a tuple of ints so I'm not sure it work because they are doing a subtraction operation on it

    @abc.abstractmethod
    def _actions(
        self, obs: EnvObs, remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]: ...

    def update_obs(self, obs: EnvObs) -> None:
        if self.memory is not None:
            self.memory.update(obs)

    def expand_obs(self, obs: EnvObs) -> EnvObs:
        if self.memory is not None:
            return self.memory.expand(obs)
        return obs

    def actions(
        self, obs: EnvObs, remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]:
        self.update_obs(obs)
        return self._actions(self.expand_obs(obs), remainingOverageTime)

    def act(
        self, step: int, obs: dict[str, Any], remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]:
        env_obs = EnvObs.from_dict(obs)

        return self.actions(env_obs, remainingOverageTime)
