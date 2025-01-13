import numpy as np
from agents.rl_agent import BasicRLAgent
from agents.models.dense import CNN
from rule_based.naive.naive_agent import NaiveAgent
from config import SAMPLING_DEVICE
from env_interface import RecordEpisode
import torch


network = CNN(23)
network.load_state_dict(
    torch.load("models_weights/network_2550.pth", weights_only=True)
)

env = RecordEpisode("records")

obs, env_params = env.reset(seed=np.random.randint(0, 1000))
agent_0 = BasicRLAgent("player_0", env_params, SAMPLING_DEVICE, network)
agent_1 = NaiveAgent("player_1", env_params)

game_finished = False
while not game_finished:
    next_obs, reward, _, truncated, _ = env.step(
        {
            "player_0": agent_0.actions(obs.player_0),
            "player_1": agent_1.actions(obs.player_1),
        }
    )
    game_finished = truncated["player_0"].item() or truncated["player_1"].item()

    obs = next_obs

env.reset()
