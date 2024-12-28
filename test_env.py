import numpy as np
from src.luxai_s3.wrappers import LuxAIS3GymEnv
from src.luxai_s3.env import Actions
import json


env = LuxAIS3GymEnv(numpy_output=True)

obs, config = env.reset()

with open("env_obs_sample.json", "w") as f:
    json.dump(obs, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

# actions: Actions = {
#     "player_0": np.zeros((16, 3), dtype=np.int32),
#     "player_1": np.zeros((16, 3), dtype=np.int32),
# }


# print(actions)

# obs, reward, terminated, truncated, infos = env.step(actions)

# print(obs["player_0"].sensor_mask)

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
