from random import randint
import numpy as np
from lux.observation import Observation
from lux.utils import Direction, print_debug
from base_agent import Agent, N_Actions


class TestAgent(Agent):
    def actions(
        self, obs: Observation, remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[int, N_Actions], np.dtype[np.int32]]:
        direction1 = Direction.UP if self.team_id == 1 else Direction.DOWN
        direction2 = Direction.LEFT if self.team_id == 1 else Direction.RIGHT

        direction = direction1 if randint(0, 1) == 0 else direction2

        if obs.step == 90:
            print_debug(obs)

        return np.array(
            [
                [
                    direction,
                    -1 if self.team_id == 1 else 0,
                    -1 if self.team_id == 1 else 0,
                ]
                for _ in range(self.env_cfg.max_units)
            ]
        )
