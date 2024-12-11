from dataclasses import dataclass


@dataclass
class EnvConfig:
    def __init__(self, env_cfg: dict[str, int]):
        self.max_units: int = env_cfg["max_units"]
        self.match_count_per_episode: int = env_cfg["match_count_per_episode"]
        self.max_steps_in_match: int = env_cfg["max_steps_in_match"]
        self.map_height: int = env_cfg["map_height"]
        self.map_width: int = env_cfg["map_width"]
        self.num_teams: int = env_cfg["num_teams"]
        self.unit_move_cost: int = env_cfg["unit_move_cost"]
        self.unit_sap_cost: int = env_cfg["unit_sap_cost"]
        self.unit_sap_range: int = env_cfg["unit_sap_range"]
        self.unit_sensor_range: int = env_cfg["unit_sensor_range"]
