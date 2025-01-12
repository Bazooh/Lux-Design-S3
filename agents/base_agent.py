from agents.lux.utils import Vector2
from agents.obs import EnvParams, Obs

from abc import ABC, abstractmethod
from typing import Any, Literal
import numpy as np
import torch


N_Players = Literal[2]
N_Actions = Literal[3]
N_Agents = Literal[16]


class Agent(ABC):
    def __init__(self, player: str, env_params: EnvParams) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_params = env_params

        self.relic_node_positions: list[Vector2] = []
        self.discovered_relic_nodes_ids: set[int] = set()
        self.relic_tensor = torch.zeros((24, 24), dtype=torch.float32)

        self.unit_explore_locations: dict[int, Any] = (
            dict()
        )  # "Any" should be "Vector2" but for some reason they put a tuple of ints so I'm not sure it work because they are doing a subtraction operation on it

    @abstractmethod
    def actions(
        self, obs: Obs, remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]: ...

    def act(
        self, step: int, obs: dict[str, Any], remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]:
        return self.actions(Obs.from_dict(obs), remainingOverageTime)
