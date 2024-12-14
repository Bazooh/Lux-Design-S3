import numpy as np
from src.luxai_s3.wrappers import LuxAIS3GymEnv
from src.luxai_s3.env import Actions


env = LuxAIS3GymEnv()

obs, _ = env.reset()


actions: Actions = {
    "player_0": np.zeros((16, 3), dtype=np.int32),
    "player_1": np.zeros((16, 3), dtype=np.int32),
}

for i in range(503):
    obs, reward, terminated, truncated, infos = env.step(actions)

obs, reward, terminated, truncated, infos = env.step(actions)
print(reward, terminated, truncated, sep='\n')

obs, reward, terminated, truncated, infos = env.step(actions)
print(reward, terminated, truncated, sep='\n')
