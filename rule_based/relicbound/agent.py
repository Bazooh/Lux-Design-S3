import sys
import os
from typing import Iterable

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import copy
import numpy as np
from sys import stderr
from scipy.signal import convolve2d

from rule_based.relicbound.utils import (
    Global,
    NodeType,
    ActionType,
    SPACE_SIZE,
    get_match_idx,
    get_match_step,
    warp_point,
    get_opposite,
    is_team_sector,
    show_map,
    show_energy_field,
    show_exploration_map,
    astar,
    find_closest_target,
    nearby_positions,
    create_weights,
    estimate_energy_cost,
    path_to_actions,
    manhattan_distance,
)


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = NodeType.unknown
        self.energy: int | None = None
        self.is_visible = False

        self._relic = False
        self._reward = False
        self._explored_for_relic = False
        self._explored_for_reward = False

    def __repr__(self):
        return f"Node({self.x}, {self.y}, {self.type})"

    def __hash__(self):
        return self.coordinates.__hash__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    @property
    def relic(self):
        return self._relic

    @property
    def reward(self):
        return self._reward

    @property
    def explored_for_relic(self):
        return self._explored_for_relic

    @property
    def explored_for_reward(self):
        return self._explored_for_reward

    def update_relic_status(self, status: bool):
        # if self._explored_for_relic and self._relic != status:
        #     raise ValueError(
        #         f"Can't change the relic status {self._relic}->{status} for {self}"
        #         ", the tile has already been explored"
        #     )

        self._relic = status
        self._explored_for_relic = True

    def update_reward_status(self, status: bool):
        # if self._explored_for_reward and self._reward != status:
        #     raise ValueError(
        #         f"Can't change the reward status {self._reward}->{status} for {self}"
        #         ", the tile has already been explored"
        #     )

        self._reward = status
        self._explored_for_reward = True

    @property
    def is_unknown(self) -> bool:
        return self.type == NodeType.unknown

    @property
    def is_walkable(self) -> bool:
        return self.type != NodeType.asteroid

    @property
    def coordinates(self) -> tuple[int, int]:
        return self.x, self.y

    def manhattan_distance(self, other: "Node") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


class Space:
    def __init__(self):
        self._nodes: list[list[Node]] = []
        for y in range(SPACE_SIZE):
            row = [Node(x, y) for x in range(SPACE_SIZE)]
            self._nodes.append(row)

        # set of nodes with a relic
        self._relic_nodes: set[Node] = set()

        # set of nodes that provide points
        self._reward_nodes: set[Node] = set()

    def __repr__(self) -> str:
        return f"Space({SPACE_SIZE}x{SPACE_SIZE})"

    def __iter__(self):
        for row in self._nodes:
            yield from row

    @property
    def relic_nodes(self) -> set[Node]:
        return self._relic_nodes

    @property
    def reward_nodes(self) -> set[Node]:
        return self._reward_nodes

    def get_node(self, x, y) -> Node:
        return self._nodes[y][x]

    def update(self, step, obs, team_id, team_reward):
        self.move_obstacles(step)
        self._update_map(obs)
        self._update_relic_map(obs, team_id, team_reward, step)

    def _update_relic_map(self, obs, team_id, team_reward, step: int):
        match_idx = get_match_idx(step)
        match_step = get_match_step(step)

        for relic_id, (mask, xy) in enumerate(
            zip(obs["relic_nodes_mask"], obs["relic_nodes"])
        ):
            if mask:
                self._update_relic_status(*xy, status=True)

        if (
            match_idx <= 2
            and match_step == 50
            and len(self._relic_nodes) != 2 * (min(match_idx, 2) + 1)
        ):
            for node in self:
                if not node.relic:
                    node._explored_for_relic = False

        all_relics_found = True
        all_rewards_found = True
        for node in self:
            if node.is_visible and not node.explored_for_relic:
                self._update_relic_status(*node.coordinates, status=False)

            if not node.explored_for_relic:
                all_relics_found = False

            if not node.explored_for_reward:
                all_rewards_found = False

        Global.ALL_RELICS_FOUND = all_relics_found
        Global.ALL_REWARDS_FOUND = all_rewards_found

        if not Global.ALL_RELICS_FOUND:
            if len(self._relic_nodes) == 2 * (min(match_idx, 2) + 1):
                # all relics found, mark all nodes as explored for relics
                Global.ALL_RELICS_FOUND = True
                for node in self:
                    if not node.explored_for_relic:
                        self._update_relic_status(*node.coordinates, status=False)

        if not Global.ALL_REWARDS_FOUND:
            self._update_reward_status_from_relics_distribution()
            self._update_reward_results(obs, team_id, team_reward)
            self._update_reward_status_from_reward_results()

    def _update_reward_status_from_reward_results(self):
        # We will use Global.REWARD_RESULTS to identify which nodes yield points
        for result in Global.REWARD_RESULTS:
            unknown_nodes = set()
            known_reward = 0
            for n in result["nodes"]:
                if not n.explored_for_reward:
                    unknown_nodes.add(n)
                    continue

                if n.reward:
                    known_reward += 1

            if not unknown_nodes:
                # all nodes already explored, nothing to do here
                continue

            reward = result["reward"] - known_reward  # reward from unknown_nodes

            if reward == 0:
                # all nodes are empty
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=False)

            elif reward == len(unknown_nodes):
                # all nodes yield points
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=True)

            # elif reward > len(unknown_nodes):
            #     # we shouldn't be here
            #     print(
            #         f"Something wrong with reward result: {result}"
            #         ", this result will be ignored.",
            #         file=stderr,
            #     )

    def _update_reward_results(self, obs, team_id, team_reward):
        ship_nodes = set()
        for active, energy, position in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                # Only units with non-negative energy can give points
                ship_nodes.add(self.get_node(*position))

        Global.REWARD_RESULTS.append({"nodes": ship_nodes, "reward": team_reward})

    def _update_reward_status_from_relics_distribution(self):
        # Rewards can only occur near relics.
        # Therefore, if there are no relics near the node
        # we can infer that the node does not contain a reward.

        relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), np.int32)
        for node in self:
            if node.relic or not node.explored_for_relic:
                relic_map[node.y][node.x] = 1

        reward_size = 2 * Global.RELIC_REWARD_RANGE + 1

        reward_map = convolve2d(
            relic_map,
            np.ones((reward_size, reward_size), dtype=np.int32),
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        for node in self:
            if reward_map[node.y][node.x] == 0:
                # no relics in range RELIC_REWARD_RANGE
                node.update_reward_status(False)

    def _reset_reward_around(self, x: int, y: int):
        for x_, y_ in nearby_positions(x, y, Global.RELIC_REWARD_RANGE):
            node = self.get_node(x_, y_)
            if not node.reward:
                node._explored_for_reward = False

                for result in Global.REWARD_RESULTS:
                    if node in result["nodes"]:
                        result["nodes"].remove(node)

    def _update_relic_status(self, x, y, status=True):
        node = self.get_node(x, y)
        node.update_relic_status(status)

        # relics are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_relic_status(status)

        if status and node not in self._relic_nodes:
            self._relic_nodes.add(node)
            self._relic_nodes.add(opp_node)
            self._reset_reward_around(x, y)
            self._reset_reward_around(*get_opposite(x, y))

    def _update_reward_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_reward_status(status)

        # rewards are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_reward_status(status)

        if status:
            self._reward_nodes.add(node)
            self._reward_nodes.add(opp_node)

    def _update_map(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_energy = obs["map_features"]["energy"]
        obs_tile_type = obs["map_features"]["tile_type"]

        obstacles_shifted = False
        energy_nodes_shifted = False
        for node in self:
            x, y = node.coordinates
            is_visible = sensor_mask[x, y]

            if (
                is_visible
                and not node.is_unknown
                and node.type.value != obs_tile_type[x, y]
            ):
                obstacles_shifted = True

            if (
                is_visible
                and node.energy is not None
                and node.energy != obs_energy[x, y]
            ):
                energy_nodes_shifted = True

        Global.OBSTACLES_MOVEMENT_STATUS.append(obstacles_shifted)

        if not Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
            period = self._find_obstacle_movement_period(
                Global.OBSTACLES_MOVEMENT_STATUS
            )
            if period is not None:
                Global.OBSTACLE_MOVEMENT_PERIOD_FOUND = True
                Global.OBSTACLE_MOVEMENT_PERIOD = period

        if not Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND and obstacles_shifted:
            direction = self._find_obstacle_movement_direction(obs)
            if direction:
                Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND = True
                Global.OBSTACLE_MOVEMENT_DIRECTION = direction

                self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)
            else:
                # Can't find OBSTACLE_MOVEMENT_DIRECTION
                for node in self:
                    node.type = NodeType.unknown

        for node in self:
            x, y = node.coordinates
            is_visible = bool(sensor_mask[x, y])

            node.is_visible = is_visible

            if is_visible and node.is_unknown:
                node.type = NodeType(int(obs_tile_type[x, y]))

                # we can also update the node type on the other side of the map
                # because the map is symmetrical
                self.get_node(*get_opposite(x, y)).type = node.type

            if is_visible:
                node.energy = int(obs_energy[x, y])

                # the energy field should be symmetrical
                self.get_node(*get_opposite(x, y)).energy = node.energy

            elif energy_nodes_shifted:
                # The energy field has changed
                # I cannot predict what the new energy field will be like.
                node.energy = None

    @staticmethod
    def _find_obstacle_movement_period(obstacles_movement_status):
        # Right now there are only two options for nebula_tile_drift_speed: 1 / 20 and 1 / 40
        if obstacles_movement_status and obstacles_movement_status[-1]:
            return 20 if len(obstacles_movement_status) - 21 % 40 < 20 else 40

    def _find_obstacle_movement_direction(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_tile_type = obs["map_features"]["tile_type"]

        suitable_directions = []
        for direction in [(1, -1), (-1, 1)]:
            moved_space = self.move(*direction, inplace=False)

            match = True
            for node in moved_space:
                x, y = node.coordinates
                if (
                    sensor_mask[x, y]
                    and not node.is_unknown
                    and obs_tile_type[x, y] != node.type.value
                ):
                    match = False
                    break

            if match:
                suitable_directions.append(direction)

        if len(suitable_directions) == 1:
            return suitable_directions[0]

    def clear(self, match_idx: int):
        for node in self:
            node.is_visible = False

    def move_obstacles(self, step):
        if (
            Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
            and Global.OBSTACLE_MOVEMENT_PERIOD > 0
            and (step - 1) % Global.OBSTACLE_MOVEMENT_PERIOD == 0
        ):
            self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)

    def move(self, dx: int, dy: int, *, inplace=False) -> "Space":
        if not inplace:
            new_space = copy.deepcopy(self)
            for node in self:
                x, y = warp_point(node.x + dx, node.y + dy)
                new_space.get_node(x, y).type = node.type
            return new_space
        else:
            types = [n.type for n in self]
            for node, node_type in zip(self, types):
                x, y = warp_point(node.x + dx, node.y + dy)
                self.get_node(x, y).type = node_type
            return self


class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.energy = 0
        self.node: Node | None = None

        self.task: str | None = None
        self.target: Node | None = None
        self.action: ActionType | None = None

    def __repr__(self):
        assert self.node is not None, "Ship is not on the map"

        return (
            f"Ship({self.unit_id}, node={self.node.coordinates}, energy={self.energy})"
        )

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    def clean(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.target = None
        self.action = None


class Fleet:
    def __init__(self, team_id):
        self.team_id: int = team_id
        self.points: int = 0  # how many points have we scored in this match so far
        self.ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.node is not None:
                yield ship

    def clear(self):
        self.points = 0
        for ship in self.ships:
            ship.clean()

    def update(self, obs, space: Space):
        self.points = int(obs["team_points"][self.team_id])

        for ship, active, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if active:
                ship.node = space.get_node(*position)
                ship.energy = int(energy)
                ship.action = None
            else:
                ship.clean()


class RelicboundAgent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

        Global.MAX_UNITS = env_cfg["max_units"]
        Global.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Global.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Global.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Global.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        match_step = get_match_step(step)
        match_idx = get_match_idx(step)

        # print(f"start step={match_step}({step})", file=stderr)

        if match_step == 0:
            # nothing to do here at the beginning of the match
            # just need to clean up some of the garbage that was left after the previous match
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear(match_idx)
            self.space.move_obstacles(step)
            return self.create_actions_array()

        points = int(obs["team_points"][self.team_id])

        # how many points did we score in the last step
        reward = max(0, points - self.fleet.points)

        self.space.update(step, obs, self.team_id, reward)
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)

        # self.show_visible_map()

        self.find_relics()
        self.harvest()
        self.find_rewards()

        # if match_idx == 0 and self.player == "player_0":
        #     for ship in self.fleet:
        #         print(f"{ship.unit_id} {ship.task} {ship.target}", file=stderr)
        #         print(Global.ALL_RELICS_FOUND, self.space.relic_nodes, file=stderr)
        #         print(Global.ALL_REWARDS_FOUND, self.space.reward_nodes, file=stderr)

        # for ship in self.fleet:
        #     print(ship, ship.task, ship.target, ship.action, file=stderr)

        return self.create_actions_array()

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)

        for i, ship in enumerate(ships):
            if ship.action is not None:
                actions[i] = ship.action, 0, 0

        return actions

    def find_relics(self):
        if Global.ALL_RELICS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_relics":
                    ship.task = None
                    ship.target = None
            return

        targets = set()
        for node in self.space:
            if not node.explored_for_relic:
                # We will only find relics in our part of the map
                # because relics are symmetrical.
                if is_team_sector(self.fleet.team_id, *node.coordinates):
                    targets.add(node.coordinates)

        def set_task(ship: Ship) -> bool:
            if ship.task and ship.task != "find_relics":
                return False

            if ship.energy < Global.UNIT_MOVE_COST:
                return False

            target, _ = find_closest_target(ship.coordinates, targets)
            if not target:
                return False

            path = astar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)
            if actions and ship.energy >= energy:
                ship.task = "find_relics"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]

                for x, y in path:
                    for xy in nearby_positions(x, y, Global.UNIT_SENSOR_RANGE):
                        if xy in targets:
                            targets.remove(xy)

                return True

            return False

        for ship in self.fleet:
            if set_task(ship):
                continue

            if ship.task == "find_relics":
                ship.task = None
                ship.target = None

    def find_rewards(self):
        if Global.ALL_REWARDS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_rewards":
                    ship.task = None
                    ship.target = None
            return

        unexplored_relics = self.get_unexplored_relics()

        if len(unexplored_relics) == 0:
            return

        def set_task(ship: Ship, target: Node):
            path = astar(create_weights(self.space), ship.coordinates, target.coordinates)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if actions and ship.energy >= energy:
                ship.task = "find_rewards"
                ship.target = target
                ship.action = actions[0]

        def nearest_node(node: Node, nodes: Iterable[Node]) -> Node:
            nearest = None
            min_distance = float("inf")
            for n in nodes:
                d = node.manhattan_distance(n)
                if d < min_distance:
                    min_distance = d
                    nearest = n

            assert nearest is not None, (
                f"nodes list is empty or as inf distance, nodes = {nodes}"
            )

            return nearest

        unexplored_points: set[Node] = set()
        for relic in unexplored_relics:
            for x, y in nearby_positions(*relic.coordinates, Global.RELIC_REWARD_RANGE):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    unexplored_points.add(node)

        rewards_ship = {
            ship
            for ship in self.fleet
            if ship.node and not ship.task or ship.task == "find_rewards"
        }

        ship_points_distance: dict[tuple[Ship, Node], int] = {}

        for ship in rewards_ship:
            if not ship.node:
                continue

            for node in unexplored_points:
                ship_points_distance[(ship, node)] = manhattan_distance(
                    ship.node.coordinates, node.coordinates
                )

        while ship_points_distance:
            ship, node = min(
                ship_points_distance, key=lambda x: ship_points_distance[x]
            )
            del ship_points_distance[(ship, node)]

            set_task(ship, node)

            for ship_points in ship_points_distance.copy():
                if ship_points[0] == ship:
                    del ship_points_distance[ship_points]
                if ship_points[1] == node:
                    del ship_points_distance[ship_points]

    def get_unexplored_relics(self) -> list[Node]:
        relic_nodes = []
        for relic_node in self.space.relic_nodes:
            if not is_team_sector(self.team_id, *relic_node.coordinates):
                continue

            explored = True
            for x, y in nearby_positions(
                *relic_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    explored = False
                    break

            if explored:
                continue

            relic_nodes.append(relic_node)

        return relic_nodes

    def harvest(self):
        def set_task(ship: Ship, target_node: Node):
            if ship.node == target_node:
                ship.task = "harvest"
                ship.target = target_node
                ship.action = ActionType.center
                return True

            path = astar(
                create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if not actions or ship.energy < energy:
                return False

            ship.task = "harvest"
            ship.target = target_node
            ship.action = actions[0]
            return True

        booked_nodes = set()
        for ship in self.fleet:
            if ship.task == "harvest":
                if ship.target is None:
                    ship.task = None
                    continue

                if set_task(ship, ship.target):
                    booked_nodes.add(ship.target)
                else:
                    ship.task = None
                    ship.target = None

        targets = set()
        for n in self.space.reward_nodes:
            if n.is_walkable and n not in booked_nodes:
                targets.add(n.coordinates)
        if not targets:
            return

        for ship in self.fleet:
            if ship.task:
                continue

            target, _ = find_closest_target(ship.coordinates, targets)

            if target and set_task(ship, self.space.get_node(*target)):
                targets.remove(target)
            else:
                ship.task = None
                ship.target = None

            if not targets:
                break

    def show_visible_energy_field(self):
        print("Visible energy field:", file=stderr)
        show_energy_field(self.space)

    def show_explored_energy_field(self):
        print("Explored energy field:", file=stderr)
        show_energy_field(self.space, only_visible=False)

    def show_visible_map(self):
        print("Visible map:", file=stderr)
        show_map(self.space, self.fleet, bool(self.opp_fleet))

    def show_explored_map(self):
        print("Explored map:", file=stderr)
        show_map(self.space, self.fleet, only_visible=False)

    def show_exploration_map(self):
        print("Exploration map:", file=stderr)
        show_exploration_map(self.space)
