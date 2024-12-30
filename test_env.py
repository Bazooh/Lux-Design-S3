import numpy as np
from env_interface import EnvInterfaceForVec
from gymnasium.vector import SyncVectorEnv


vec_env = SyncVectorEnv([lambda: EnvInterfaceForVec() for _ in range(3)])

vec_env.reset()

print(vec_env.step([np.zeros((2, 16, 3), dtype=np.int32) for _ in range(3)]))
