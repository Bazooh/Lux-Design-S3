from agents.lux.utils import direction_to, Vector2
import numpy as np
from agents.base_agent import Agent, N_Actions, N_Agents
from luxai_s3.state import EnvObs


class NaiveAgent(Agent):
    def actions(
        self, obs: EnvObs, remainingOverageTime: int
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]:
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        actions = np.zeros((self.env_cfg.max_units, 3), dtype=np.int32)

        # basic strategy here is simply to have some units randomly explore and some units collecting as much energy as possible
        # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
        # and information about where relic nodes are found are saved for the next match

        # save any new relic nodes that we discover for the rest of the game.
        for id in obs.relics.avaible_ids():
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(obs.relics.positions[id])

        # unit ids range from 0 to max_units - 1
        for unit_id in obs.allied_units.avaible_ids():
            unit_pos: Vector2 = obs.allied_units.positions[unit_id]

            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
                manhattan_distance: int = abs(
                    unit_pos[0] - nearest_relic_node_position[0]
                ) + abs(unit_pos[1] - nearest_relic_node_position[1])

                # if close to the relic node we want to hover around it and hope to gain points
                if manhattan_distance <= 4:
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    # otherwise we want to move towards the relic node
                    actions[unit_id] = [
                        direction_to(unit_pos, nearest_relic_node_position),
                        0,
                        0,
                    ]
            else:
                # randomly explore by picking a random location on the map and moving there for about 20 steps
                if obs.step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (
                        np.random.randint(0, self.env_cfg.map_width),
                        np.random.randint(0, self.env_cfg.map_height),
                    )
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [
                    direction_to(unit_pos, self.unit_explore_locations[unit_id]),
                    0,
                    0,
                ]

        return actions
