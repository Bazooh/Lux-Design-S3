from agents.obs import Obs
import agents.tensor_converters.channel as c
import agents.tensor_converters.raw_input as r

import numpy as np


class TensorConverter:
    def __init__(self, *infos: c.TensorChannels | r.TensorRawInputs):
        self.channels = [
            channel for channel in infos if isinstance(channel, c.TensorChannels)
        ]
        self.raw_inputs = [
            raw_input for raw_input in infos if isinstance(raw_input, r.TensorRawInputs)
        ]

    def n_channels(self) -> int:
        return sum(channel.n_channels for channel in self.channels)

    def n_raw_inputs(self) -> int:
        return sum(raw_input.n_channels for raw_input in self.raw_inputs)

    def channel_names(self) -> list[str]:
        return [name for channel in self.channels for name in channel.names]

    def raw_input_names(self) -> list[str]:
        return [name for raw_input in self.raw_inputs for name in raw_input.names]

    def update_memory(self, obs: Obs, team_id: int) -> None:
        for channel in self.channels:
            channel.update_memory(obs, team_id)
        for raw_input in self.raw_inputs:
            raw_input.update_memory(obs, team_id)

    def reset_memory(self) -> None:
        for channel in self.channels:
            channel.reset_memory()
        for raw_input in self.raw_inputs:
            raw_input.reset_memory()

    def convert_channels(self, obs: Obs, teams_id: int) -> np.ndarray:
        """Convert an observation into a tensor representation."""
        return np.concatenate(
            [channel.convert(obs, teams_id) for channel in self.channels]
        )

    def convert_raw_inputs(self, obs: Obs, teams_id: int) -> np.ndarray:
        return np.concatenate(
            [raw_input.convert(obs, teams_id) for raw_input in self.raw_inputs]
        )


class BasicTensorConverter(TensorConverter):
    def __init__(self):
        super().__init__(
            c.SensorChannel(),
            c.AsteroidChannel(),
            c.NebulaChannel(),
            c.EnergyChannel(),
            c.EnemiesChannel(),
            c.AllyUnitsChannels(),
            c.RelicPointsChannels(),
            r.StepsLeftRawInput(),
            r.MatchStepsLeftRawInput(),
            r.PointsRawInput(),
            r.UnitsEnergyRawInput(),
        )
