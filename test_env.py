import numpy as np
from src.luxai_s3.wrappers import LuxAIS3GymEnv
from src.luxai_s3.env import Actions


env = LuxAIS3GymEnv()

obs, config = env.reset()

print(obs["player_0"].sensor_mask.sum())

actions: Actions = {
    "player_0": np.zeros((16, 3), dtype=np.int32),
    "player_1": np.zeros((16, 3), dtype=np.int32),
}


# print(actions)

obs, reward, terminated, truncated, infos = env.step(actions)

print(obs["player_0"].sensor_mask.sum())

# for i in range(503):
#     actions: Actions = {
#         "player_0": agent_0.actions(obs["player_0"]),
#         "player_1": agent_1.actions(obs["player_1"]),
#     }
#     obs, reward, terminated, truncated, infos = env.step(actions)

#     print(obs["player_0"].units_mask[0][:5])

# obs, reward, terminated, truncated, infos = env.step(actions)
# print(reward, terminated, truncated, sep='\n')

# obs, reward, terminated, truncated, infos = env.step(actions)
# print(reward, terminated, truncated, sep='\n')
