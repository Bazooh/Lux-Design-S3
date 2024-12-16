import numpy as np
from src.luxai_s3.wrappers import LuxAIS3GymEnv
from src.luxai_s3.env import Actions

from agents.rl_agent import BasicRLAgent as Agent


env = LuxAIS3GymEnv()

obs, config = env.reset()

print(type(obs["player_0"].units.position))


agent_0 = Agent(
    "player_0", config["params"], model_path="models_weights/network_8100.pth"
)
agent_1 = Agent(
    "player_1", config["params"], model_path="models_weights/network_8100.pth"
)

actions_0 = agent_0.actions(obs["player_0"])
actions_1 = agent_1.actions(obs["player_1"])

actions: Actions = {
    "player_0": actions_0,
    "player_1": actions_1,
}

# actions: Actions = {
#     "player_0": np.zeros((16, 3), dtype=np.int32),
#     "player_1": np.zeros((16, 3), dtype=np.int32),
# }

print(actions)

obs, reward, terminated, truncated, infos = env.step(actions)

# for i in range(503):
#     actions: Actions = {
#         "player_0": agent_0.actions(obs["player_0"]),
#         "player_1": agent_1.actions(obs["player_1"]),
#     }
#     obs, reward, terminated, truncated, infos = env.step(actions)

# obs, reward, terminated, truncated, infos = env.step(actions)
# print(reward, terminated, truncated, sep='\n')

# obs, reward, terminated, truncated, infos = env.step(actions)
# print(reward, terminated, truncated, sep='\n')
