import abc
from typing import Any, Literal
from agents.lux.utils import Vector2, print_debug
from agents.lux.observation import Observation
from agents.lux.env_config import EnvConfig
import numpy as np


N_Actions = Literal[3]


class Agent:
    def __init__(self, player: str, env_cfg: dict[str, int]) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = EnvConfig(env_cfg)

        self.relic_node_positions: list[Vector2] = []
        self.discovered_relic_nodes_ids: set[int] = set()
        self.unit_explore_locations: dict[int, Any] = (
            dict()
        )  # "Any" should be "Vector2" but for some reason they put a tuple of ints so I'm not sure it work because they are doing a subtraction operation on it

    @abc.abstractmethod
    def actions(
        self, obs: Observation, remainingOverageTime: int
    ) -> np.ndarray[tuple[int, N_Actions], np.dtype[np.int32]]: ...

    def act(
        self, step: int, obs: dict[str, Any], remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[int, N_Actions], np.dtype[np.int32]]:
        print_debug(obs)
        return self.actions(Observation(obs, self.team_id, step), remainingOverageTime)
