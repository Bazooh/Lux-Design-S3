from agents.obs import Obs
import agents.tensor_converters.channel as c

import numpy as np


class TensorConverter:
    def __init__(self, *channels: c.TensorChannels):
        self.channels = channels

    def n_channels(self) -> int:
        return sum(channel.n_channels for channel in self.channels)

    def channel_names(self) -> list[str]:
        return [name for channel in self.channels for name in channel.names]

    def update_memory(self, obs: Obs, team_id: int) -> None:
        for channel in self.channels:
            channel.update_memory(obs, team_id)

    def reset_memory(self) -> None:
        for channel in self.channels:
            channel.reset_memory()

    def convert(self, obs: Obs, teams_id: int) -> np.ndarray:
        """Convert an observation into a tensor representation."""
        return np.concatenate(
            [channel.convert(obs, teams_id) for channel in self.channels]
        )


class BasicTensorConverter(TensorConverter):
    def __init__(self):
        super().__init__(
            c.SensorChannel(),
            c.AsteroidChannel(),
            c.NebulaChannel(),
            c.EnergyChannel(),
            c.EnemiesChannel(),
            c.RelicPointsChannels(),
            c.AllyUnitsChannels(),
        )
