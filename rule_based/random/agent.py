import numpy as np
from agents.base_agent import Agent, N_Actions, N_Agents
from agents.obs import Obs


class RandomAgent(Agent):
    def _actions(
        self, obs: Obs, remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]:
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        actions = np.zeros((16, 3), dtype=np.int32)

        # unit ids range from 0 to max_units - 1
        for unit_id in obs.get_available_units(self.team_id):
            actions[unit_id] = [
                np.random.randint(0, 5),
                0,
                0,
            ]

        return actions
