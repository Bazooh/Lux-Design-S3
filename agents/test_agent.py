from typing import Any
from lux.utils import direction_to, Vector2, print_debug
import numpy as np
from base_agent import Agent, N_Actions


class TestAgent(Agent):
    def act(
        self, step: int, obs: dict[str, Any], remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[int, N_Actions], np.dtype[np.int32]]:
        # print_debug(f"Observation: {obs}")
        return np.array([[0, 0, 0] for _ in range(self.env_cfg["max_units"])])
