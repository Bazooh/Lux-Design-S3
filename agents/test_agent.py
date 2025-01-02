from random import randint
import numpy as np
from agents.lux.utils import Direction
from agents.base_agent import Agent, N_Actions, N_Agents
from agents.obs import Obs


class TestAgent(Agent):
    def _actions(
        self, obs: Obs, remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]:
        direction1 = Direction.UP if self.team_id == 1 else Direction.DOWN
        direction2 = Direction.LEFT if self.team_id == 1 else Direction.RIGHT

        direction = direction1 if randint(0, 1) == 0 else direction2

        return np.array(
            [
                [
                    direction,
                    -1 if self.team_id == 1 else 0,
                    -1 if self.team_id == 1 else 0,
                ]
                for _ in range(self.env_params.max_units)
            ]
        )
